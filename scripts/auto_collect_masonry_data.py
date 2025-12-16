#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动化采集masonry insertion任务数据的脚本
使用基于位置的简单控制器自动执行pick和insertion任务，替代人工操作
完全复用gym_manipulator的数据采集流程，只是用自动控制器替换gamepad输入
"""

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import av

# 添加lerobot路径
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot" / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.processor import (
    TransitionKey,
    create_transition,
)
from lerobot.rl.acfql.gym_manipulator import (
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
    get_frequency_stats,
)
from lerobot.utils.utils import init_logging, TimerManager, log_say
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.import_utils import register_third_party_devices

# 注册第三方设备（必须在导入gym_hil之前）
register_third_party_devices()

# 导入gym_hil wrapper
from gym_hil.wrappers.hil_wrappers import (
    EEActionWrapper,
    GripperPenaltyWrapper,
    ResetDelayWrapper,
    DEFAULT_EE_STEP_SIZE,
)
from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper

# 初始化日志
init_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomaticMasonryController:
    """自动控制器，用于masonry insertion任务
    模拟人类操作：pick brick -> lift -> move to target -> place -> release
    """
    
    def __init__(self, env):
        self.env = env
        # 找到底层环境（unwrapped），用于访问MuJoCo数据
        self.base_env = env
        while hasattr(self.base_env, 'env'):
            self.base_env = self.base_env.env
        # 如果还是找不到，尝试unwrapped属性
        if not hasattr(self.base_env, '_data') and hasattr(self.base_env, 'unwrapped'):
            self.base_env = self.base_env.unwrapped
        
        self.phase = "approach_block"  # approach_block, grasp, lift, move_to_target, place, release
        # 目标位置将从环境动态获取（在reset时初始化），避免硬编码
        self.target_pos = None  # 将在reset()时从环境获取
        self.grasp_step = 0
        self.place_step = 0
        self.release_step = 0
        self.last_block_pos = None
        self.block_stable_steps = 0
        self.initial_block_z = None  # 记录初始block Z位置，用于判断是否成功抓取
        
        # 验证是否能访问底层环境
        if hasattr(self.base_env, '_data'):
            logger.info("  [Controller] ✅ 成功访问底层环境，可以获取MuJoCo数据")
        else:
            logger.warning("  [Controller] ⚠️  无法访问底层环境，可能无法正确获取状态")
    
    def get_gripper_ctrl_value(self):
        """获取gripper当前的控制值（0-255）"""
        try:
            if hasattr(self.base_env, '_gripper_ctrl_id') and self.base_env._gripper_ctrl_id is not None:
                return float(self.base_env._data.ctrl[self.base_env._gripper_ctrl_id])
            elif hasattr(self.base_env, '_data') and hasattr(self.base_env, '_model'):
                # 尝试查找fingers_actuator
                try:
                    import mujoco
                    actuator_id = mujoco.mj_name2id(self.base_env._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
                    if actuator_id >= 0:
                        return float(self.base_env._data.ctrl[actuator_id])
                except Exception:
                    pass
        except Exception:
            pass
        return None  # 无法获取
    
    def get_gripper_joint_angles(self):
        """获取gripper的joint角度（用于调试可视化问题）"""
        try:
            # 获取right_driver_joint和left_driver_joint的角度
            right_joint_id = self.base_env._model.joint("right_driver_joint").id
            left_joint_id = self.base_env._model.joint("left_driver_joint").id
            right_angle = float(self.base_env._data.qpos[right_joint_id])
            left_angle = float(self.base_env._data.qpos[left_joint_id])
            return right_angle, left_angle
        except Exception:
            return None, None
        
    def reset(self):
        """重置控制器状态"""
        self.phase = "approach_block"
        self.grasp_step = 0
        self.place_step = 0
        self.release_step = 0
        self.last_block_pos = None
        self.block_stable_steps = 0
        self.initial_block_z = None  # 重置初始block Z位置
        
        # 从环境获取target位置（动态获取，避免硬编码）
        try:
            if hasattr(self.base_env, '_get_target_pos'):
                self.target_pos = self.base_env._get_target_pos().copy()
                logger.info(f"  [Controller] ✅ 从环境获取target位置: {self.target_pos}")
            elif hasattr(self.base_env, '_TARGET_POS'):
                self.target_pos = self.base_env._TARGET_POS.copy()
                logger.info(f"  [Controller] ✅ 从环境获取target位置: {self.target_pos}")
            else:
                # Fallback: 使用默认值（应该与当前环境配置一致）
                self.target_pos = np.array([0.6, 0.0, 0.362])  # Foundation 3cm，第4层中心Z=0.362m
                logger.warning(f"  [Controller] ⚠️  无法从环境获取target位置，使用默认值: {self.target_pos}")
        except Exception as e:
            # Fallback: 使用默认值
            self.target_pos = np.array([0.6, 0.0, 0.362])  # Foundation 3cm，第4层中心Z=0.362m
            logger.warning(f"  [Controller] ⚠️  获取target位置失败 ({e})，使用默认值: {self.target_pos}")
        
        # 重置日志计数器
        self._last_log_step = 0
        self._lift_log_step = 0
        self._move_log_step = 0
        # 重置lift相关变量
        self._lift_start_ee_z = None
        self._lift_start_block_z = None
        # 重置grasp等待计时器
        self._grasp_close_wait_start = None
        # 重置警告标志
        self._warned_block_pos = False
        self._warned_ee_pos = False
        self._error_logged = False
        # 重置release相关变量
        if hasattr(self, 'release_step'):
            self.release_step = 0
        # 清除碰撞检测状态
        if hasattr(self, '_last_block_target_dist'):
            delattr(self, '_last_block_target_dist')
        logger.info(f"  [Controller] 🔄 重置控制器，初始阶段: {self.phase}")
        
    def get_action(self):
        """根据当前环境状态生成动作 [delta_x, delta_y, delta_z, gripper]"""
        block_pos = None
        ee_pos = None
        
        try:
            # 从底层环境获取block位置（使用sensor）
            if hasattr(self.base_env, '_data'):
                try:
                    block_pos = self.base_env._data.sensor("block_pos").data.copy()
                except Exception as e:
                    logger.debug(f"Failed to get block_pos: {e}")
                
                # 获取end-effector位置（优先使用pinch_pos sensor，这是gripper的中心点）
                try:
                    # 尝试使用sensor "2f85/pinch_pos"
                    ee_pos = self.base_env._data.sensor("2f85/pinch_pos").data.copy()
                except Exception as e1:
                    # Fallback: 尝试从site获取
                    try:
                        if hasattr(self.base_env, '_model'):
                            import mujoco
                            # 尝试不同的site名称
                            for site_name in ["pinch", "2f85/pinch", "ee", "end_effector"]:
                                try:
                                    site_id = mujoco.mj_name2id(self.base_env._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                                    if site_id >= 0:
                                        ee_pos = self.base_env._data.site_xpos[site_id].copy()
                                        break
                                except:
                                    continue
                    except Exception as e2:
                        pass
                    
                    # 最后的fallback: 使用ee_site_id
                    if ee_pos is None and hasattr(self.base_env, '_ee_site_id') and self.base_env._ee_site_id is not None:
                        ee_pos = self.base_env._data.site_xpos[self.base_env._ee_site_id].copy()
        except Exception as e:
            if not hasattr(self, '_error_logged'):
                logger.warning(f"Error getting positions: {e}")
                self._error_logged = True
        
        if block_pos is None:
            block_pos = np.array([0.5, 0.0, 0.06])  # 默认block位置
            if not hasattr(self, '_warned_block_pos'):
                logger.warning("Using default block_pos - 无法获取环境状态")
                self._warned_block_pos = True
        
        if ee_pos is None:
            ee_pos = np.array([0.4, 0.0, 0.3])  # 默认EE位置
            if not hasattr(self, '_warned_ee_pos'):
                logger.warning("Using default ee_pos - 无法获取环境状态")
                self._warned_ee_pos = True
        
        # 记录初始block Z位置（在第一次调用时，或者在reset后第一次获取时）
        # 如果block已经掉到地面上（Z < 0.05m），说明之前的抓取失败了，需要重新记录初始位置
        if self.initial_block_z is None:
            self.initial_block_z = block_pos[2]
        elif block_pos[2] < 0.05 and self.initial_block_z > 0.05:
            # Block掉到地面上了，更新初始Z为当前值（地面上的block中心位置）
            logger.warning(f"  [Controller] Block已掉到地面上 (block_z={block_pos[2]:.3f})，更新initial_block_z")
            self.initial_block_z = block_pos[2]
        
        # 检测block是否稳定（用于判断是否成功抓取）
        if self.last_block_pos is not None:
            block_moved = np.linalg.norm(block_pos - self.last_block_pos)
            if block_moved < 0.005:  # 移动小于5mm
                self.block_stable_steps += 1
            else:
                self.block_stable_steps = 0
        self.last_block_pos = block_pos.copy()
        
        # 获取gripper当前状态
        gripper_ctrl_value = self.get_gripper_ctrl_value()
        gripper_normalized = (gripper_ctrl_value / 255.0) if gripper_ctrl_value is not None else None  # MAX_GRIPPER_COMMAND = 255
        # 获取gripper joint角度（用于调试可视化问题）
        try:
            right_angle, left_angle = self.get_gripper_joint_angles()
        except:
            right_angle, left_angle = None, None
        
        # 计算delta动作 [delta_x, delta_y, delta_z, gripper]
        action = np.zeros(4, dtype=np.float32)
        step_size = 0.025  # 每步移动2.5cm（与EEActionWrapper一致）
        
        # 根据阶段执行不同策略
        if self.phase == "approach_block":
            # 阶段1: 移动到block上方（分步接近，避免碰撞）
            # 策略：先XY对齐，再垂直下降，避免斜向移动时gripper撞到block
            dist_xy = np.linalg.norm(ee_pos[:2] - block_pos[:2])
            dist_z = ee_pos[2] - block_pos[2]
            
            if dist_xy > 0.03:  # XY距离>3cm，先水平移动到block正上方
                # 只移动XY，保持Z高度（至少高于block 18cm，确保安全距离）
                target_xy = block_pos[:2].copy()
                safe_height = max(block_pos[2] + 0.18, ee_pos[2])  # 至少高于block 18cm
                target_above_block = np.array([target_xy[0], target_xy[1], safe_height])
                # 使用较小的步长水平移动，更平滑
                xy_step_size = 0.02  # 2cm步长
                delta_xy = target_above_block[:2] - ee_pos[:2]
                delta_xy = np.clip(delta_xy, -xy_step_size, xy_step_size)
                delta = np.array([delta_xy[0], delta_xy[1], 0])  # Z方向不移动
                action[:3] = np.concatenate([delta_xy / xy_step_size, [0]])  # Z方向归一化到0
            elif dist_z > 0.12:  # XY已对齐，但Z太高，垂直下降到安全高度
                # 垂直下降，只移动Z方向，保持XY不变
                target_above_block = block_pos.copy()
                target_above_block[2] += 0.12  # 下降到block上方12cm
                z_step_size = 0.02  # 2cm步长，更平滑
                delta_z = target_above_block[2] - ee_pos[2]
                delta_z = np.clip(delta_z, -z_step_size, z_step_size)
                delta = np.array([0, 0, delta_z])  # 只移动Z方向
                action[:3] = np.array([0, 0, delta_z / z_step_size])
            else:  # 已经很接近了
                target_above_block = block_pos.copy()
                target_above_block[2] += 0.12
                delta = target_above_block - ee_pos
                delta = np.clip(delta, -step_size, step_size)
                action[:3] = delta / step_size
            
            action[3] = 0.0  # gripper打开 (范围[0,2]: 0=打开, 2=关闭, 1=中性) [已反转：0=打开, 2=关闭]
            
            # 调试输出（每10步打印一次）
            if not hasattr(self, '_last_log_step') or self._last_log_step % 10 == 0:
                dist_to_target = np.linalg.norm(delta)
                logger.info(f"  [Controller] {self.phase}: block={block_pos}, ee={ee_pos}, "
                          f"target_above={target_above_block}, delta={delta}, dist={dist_to_target:.3f}m, "
                          f"dist_xy={dist_xy:.3f}m, dist_z={dist_z:.3f}m")
                self._last_log_step = 0
            self._last_log_step = getattr(self, '_last_log_step', 0) + 1
            
            # 如果接近block上方（XY<3cm且Z在12cm±2cm范围内），进入抓取阶段
            if dist_xy < 0.03 and abs(dist_z - 0.12) < 0.02:
                self.phase = "grasp"
                self.grasp_step = 0
                logger.info(f"  [Controller] ✅ 阶段转换: approach_block -> grasp "
                          f"(dist_xy={dist_xy:.3f}m, dist_z={dist_z:.3f}m)")
                
        elif self.phase == "grasp":
            # 阶段2: 抓取brick中心位置
            # 策略：先定位（XY精确对齐到brick中心上方）→ 后下降（垂直下降到brick中心高度）→ 最后抓取（关闭gripper）
            # block_pos是砖块的几何中心位置，直接使用它作为抓取目标
            target_block_center = block_pos.copy()  # brick中心位置
            
            # 计算XY和Z距离
            dist_xy = np.linalg.norm(ee_pos[:2] - block_pos[:2])
            dist_z = abs(ee_pos[2] - block_pos[2])
            dist_3d = np.linalg.norm(block_pos - ee_pos)
            
            wrapper_step_size = 0.025  # EEActionWrapper的默认step_size
            
            # 子阶段1: XY精确定位到brick中心上方（先不下降Z）
            if dist_xy > 0.005:  # XY距离>5mm，先精确定位XY
                # 只移动XY，保持Z不变
                target_above_block = np.array([block_pos[0], block_pos[1], ee_pos[2]])
                error_xy = block_pos[:2] - ee_pos[:2]
                
                # 根据距离自适应步长
                if dist_xy > 0.05:  # 距离>5cm
                    xy_step_size = 0.02  # 2cm步长
                elif dist_xy > 0.02:  # 距离2-5cm
                    xy_step_size = 0.015  # 1.5cm步长
                else:  # 距离<2cm，精细调整
                    xy_step_size = 0.01  # 1cm步长
                
                delta_xy = np.clip(error_xy, -xy_step_size, xy_step_size)
                action_xy = delta_xy / wrapper_step_size
                action[:3] = np.array([action_xy[0], action_xy[1], 0])  # Z=0，不下降
                action[:3] = np.clip(action[:3], -1.0, 1.0)
                action[3] = 0.0  # 保持gripper打开，等待定位完成
                
            # 子阶段2: XY已对齐，垂直下降到brick中心高度
            elif dist_z > 0.01:  # XY已对齐（<5mm），但Z距离>1cm，垂直下降
                # 只移动Z方向，保持XY不变
                error_z = block_pos[2] - ee_pos[2]
                
                # 根据距离自适应步长
                if dist_z > 0.05:  # 距离>5cm
                    z_step_size = 0.02  # 2cm步长
                elif dist_z > 0.02:  # 距离2-5cm
                    z_step_size = 0.015  # 1.5cm步长
                else:  # 距离<2cm，精细调整
                    z_step_size = 0.01  # 1cm步长
                
                delta_z_clipped = np.clip(error_z, -z_step_size, z_step_size)
                action_z = delta_z_clipped / wrapper_step_size
                action[:3] = np.array([0, 0, action_z])  # XY=0，只下降Z
                action[:3] = np.clip(action[:3], -1.0, 1.0)
                
                # 当接近brick（Z距离<3cm）时，开始关闭gripper
                if dist_z < 0.03:
                    action[3] = 2.0  # 开始关闭gripper
                else:
                    action[3] = 0.0  # 保持打开，继续下降
                    
            # 子阶段3: XY和Z都已对齐，关闭gripper并等待抓取稳定
            else:  # XY和Z都已对齐（<5mm和<1cm），关闭gripper并等待
                action[:3] = [0, 0, 0]  # 停止移动，保持位置
                action[3] = 2.0  # 关闭gripper
            
            self.grasp_step += 1
            
            # 调试输出
            if self.grasp_step % 5 == 0:
                dist_ee_block = np.linalg.norm(block_pos - ee_pos)
                # 确定当前子阶段
                if dist_xy > 0.005:
                    sub_phase = "定位XY"
                elif dist_z > 0.01:
                    sub_phase = "下降Z"
                else:
                    sub_phase = "关闭gripper"
                gripper_status_str = "打开" if action[3] < 0.5 else "关闭"
                if gripper_ctrl_value is not None:
                    gripper_status_str += f" (ctrl={gripper_ctrl_value:.0f}, normalized={gripper_normalized:.2f}, {'关闭' if gripper_normalized < 0.1 else '打开' if gripper_normalized > 0.9 else '中间'})"
                if right_angle is not None and left_angle is not None:
                    gripper_status_str += f" [joints: R={right_angle:.3f}, L={left_angle:.3f}]"
                logger.info(f"  [Controller] {self.phase} ({sub_phase}, step={self.grasp_step}): "
                          f"target={target_block_center}, ee={ee_pos}, block={block_pos}, "
                          f"dist_xy={dist_xy*1000:.1f}mm, dist_z={dist_z*1000:.1f}mm, dist_3d={dist_3d*1000:.1f}mm, "
                          f"gripper={gripper_status_str}")
            
            # 检查是否成功抓取并可以进入lift阶段
            dist_ee_block = np.linalg.norm(block_pos - ee_pos)
            
            # 进入lift的条件：
            # 1. XY和Z都已对齐（<5mm和<1cm）
            # 2. 已关闭gripper并等待足够时间（至少5步，约0.5秒）
            # 3. Block和EE的距离足够近（<3cm），表示已经抓住
            xy_aligned = dist_xy < 0.005  # XY对齐到5mm以内
            z_aligned = dist_z < 0.01     # Z对齐到1cm以内
            
            # 记录何时开始关闭gripper（当XY和Z都已对齐时）
            if xy_aligned and z_aligned:
                if not hasattr(self, '_grasp_close_wait_start') or self._grasp_close_wait_start is None:
                    self._grasp_close_wait_start = self.grasp_step
                waited_at_close = (self.grasp_step >= self._grasp_close_wait_start + 5)  # 等待5步（约0.5秒）
            else:
                waited_at_close = False
            
            # 可以进入lift的条件
            if xy_aligned and z_aligned and waited_at_close and dist_ee_block < 0.03:
                wait_time = (self.grasp_step - self._grasp_close_wait_start) if hasattr(self, '_grasp_close_wait_start') and self._grasp_close_wait_start is not None else 0
                self.phase = "lift"
                # 重置lift相关变量
                self._lift_log_step = 0
                self._lift_start_ee_z = None
                self._lift_start_block_z = None
                self._grasp_close_wait_start = None
                logger.info(f"  [Controller] ✅ 阶段转换: grasp -> lift (grasp_step={self.grasp_step}, "
                          f"dist_xy={dist_xy*1000:.1f}mm, dist_z={dist_z*1000:.1f}mm, dist_ee_block={dist_ee_block*1000:.1f}mm, "
                          f"等待时间={wait_time}步)")
            elif self.grasp_step > 40:  # 减少超时时间到40步（约4秒）
                # 超时保护：如果40步后还没完成，强制进入lift阶段
                logger.warning(f"  [Controller] ⚠️  Grasp超时（{self.grasp_step}步），强制进入lift阶段")
                self.phase = "lift"
                # 重置lift相关变量
                self._lift_log_step = 0
                self._lift_start_ee_z = None
                self._lift_start_block_z = None
                self._grasp_close_wait_start = None
                
        elif self.phase == "lift":
            # 阶段3: 向上提升block（保持gripper关闭）
            # 先缓慢提升，观察block是否跟随
            # 确保在第一次进入lift阶段时初始化这些变量
            if not hasattr(self, '_lift_log_step') or self._lift_log_step == 0:
                self._lift_log_step = 0
                self._lift_start_ee_z = ee_pos[2]  # 记录lift开始时的EE Z位置
                self._lift_start_block_z = block_pos[2]  # 记录lift开始时的block Z位置
                logger.info(f"  [Controller] {self.phase}: 初始化lift阶段 (ee_z={self._lift_start_ee_z:.3f}, block_z={self._lift_start_block_z:.3f})")
            
            self._lift_log_step += 1
            
            # 前5步：保持位置，确保gripper完全关闭并施加力
            if self._lift_log_step <= 5:
                action[:3] = [0, 0, 0]  # 保持位置不动
                action[3] = 2.0  # 保持关闭（0.0=打开, 2.0=关闭）
                logger.info(f"  [Controller] {self.phase}: 等待gripper完全关闭并施加力 (step={self._lift_log_step}/5)")
            else:
                # 缓慢向上提升（减小步长，更稳定）
                slow_step_size = 0.015  # 减小步长到1.5cm
                target_lifted = ee_pos.copy()
                target_lifted[2] += 0.18  # 提升18cm
                delta = target_lifted - ee_pos
                delta = np.clip(delta, -slow_step_size, slow_step_size)
                
                action[:3] = delta / slow_step_size  # 归一化到[-1, 1]
                action[3] = 2.0  # 保持关闭 (范围[0,2]: 0=打开, 2=关闭)
            
            # 检查block是否跟随EE移动（验证抓取成功）
            dist_ee_block = np.linalg.norm(block_pos - ee_pos)
            
            # 确保在计算lift amount之前，_lift_start_ee_z和_lift_start_block_z已经初始化
            if self._lift_start_ee_z is None or self._lift_start_block_z is None:
                self._lift_start_ee_z = ee_pos[2]
                self._lift_start_block_z = block_pos[2]
                logger.info(f"  [Controller] {self.phase}: 延迟初始化lift阶段 (ee_z={self._lift_start_ee_z:.3f}, block_z={self._lift_start_block_z:.3f})")
            
            # ⚠️ 关键优化：立即检查EE和Block之间的距离，如果距离过大，立即重试，无需等待
            # 这是衡量抓取稳固性的最直接指标，比Z轴变化更快地反映了滑脱
            if dist_ee_block > 0.06 and self._lift_log_step > 5:
                # 距离超过6cm且尝试了5步，说明抓取失败，立即重试
                logger.warning(f"  [Controller] ⚠️  抓取失败！Block与EE距离过大 ({dist_ee_block:.3f}m > 0.06m)，重新尝试抓取")
                # 如果block掉到地面上，需要先打开gripper，然后重新approach
                if block_pos[2] < 0.05:
                    logger.info(f"  [Controller] Block在地面上，先打开gripper，然后重新approach")
                    self.phase = "approach_block"  # 重新从approach开始
                    # 更新initial_block_z为当前值（地面上的位置）
                    self.initial_block_z = block_pos[2]
                    action[3] = 0.0  # 打开gripper (0.0=打开, 2.0=关闭)
                else:
                    self.phase = "grasp"  # 如果block还在空中，直接从grasp重试
                self.grasp_step = 0
                # 重置lift相关变量
                self._lift_log_step = 0
                self._lift_start_ee_z = None
                self._lift_start_block_z = None
                self._grasp_close_wait_start = None
                return action.astype(np.float32)  # 立即返回，避免继续执行后续逻辑
            
            # 检查block是否真的被提升了（block Z位置上升）
            # 注意：如果initial_block_z < 0.05（block在地面上），只要block_z > initial_block_z就说明被提升了
            block_lifted = False
            if self.initial_block_z is not None:
                if self.initial_block_z < 0.05:
                    # Block原本在地面上，只要Z上升了就算提升（降低阈值到0.005m，即5mm）
                    block_lifted = block_pos[2] > self.initial_block_z + 0.005  # 至少提升5mm
                else:
                    # Block原本在空中，需要提升至少1cm
                    block_lifted = block_pos[2] > self.initial_block_z + 0.01  # block提升了至少1cm
            
            # 检查block是否跟随EE移动（EE上升时，block也应该上升）
            # 计算从lift开始到现在，EE和block各自上升了多少
            ee_lifted_amount = ee_pos[2] - self._lift_start_ee_z
            block_lifted_amount = block_pos[2] - self._lift_start_block_z
            
            # Block跟随判断：block应该跟随EE上升（至少上升EE上升量的30%）
            # 但首先检查距离（更直接的指标）
            block_following = (dist_ee_block < 0.08) and (block_lifted_amount > 0) and (block_lifted_amount > ee_lifted_amount * 0.3)
            
            # 调试输出
            if self._lift_log_step % 5 == 0:
                initial_z_str = f"{self.initial_block_z:.3f}" if self.initial_block_z is not None else "None"
                logger.info(f"  [Controller] {self.phase}: ee_z={ee_pos[2]:.3f}, block_z={block_pos[2]:.3f}, "
                          f"initial_z={initial_z_str}, "
                          f"ee_block_dist={dist_ee_block:.3f}m, block_lifted={block_lifted}, "
                          f"ee_lifted={ee_lifted_amount:.3f}m, block_lifted={block_lifted_amount:.3f}m, "
                          f"following={block_following}")
            
            # 如果block被成功提升，移动到target
            # 对于在地面上的block，提升到0.08m就足够了；对于原本在空中的block，需要提升3cm
            block_lifted_enough = False
            if self.initial_block_z is not None:
                if self.initial_block_z < 0.05:
                    # Block原本在地面上，提升到0.08m以上就算足够（降低要求）
                    block_lifted_enough = block_pos[2] > 0.08
                else:
                    # Block原本在空中，需要从初始位置提升3cm（降低要求）
                    block_lifted_enough = block_pos[2] > self.initial_block_z + 0.03
            
            if block_lifted_enough and block_following:
                self.phase = "move_to_target"
                initial_z_str = f"{self.initial_block_z:.3f}" if self.initial_block_z is not None else "None"
                logger.info(f"  [Controller] ✅ 阶段转换: lift -> move_to_target "
                          f"(ee_z={ee_pos[2]:.3f}, block_z={block_pos[2]:.3f}, "
                          f"initial_z={initial_z_str}, lifted={block_lifted_enough}, "
                          f"ee_block_dist={dist_ee_block:.3f}m, following={block_following})")
                # 重置lift相关变量
                self._lift_log_step = 0
                self._lift_start_ee_z = None
                self._lift_start_block_z = None
            elif self._lift_log_step > 20 and (not block_lifted or not block_following):
                # 如果20步后block还没有被提升或跟随（说明抓取失败），重新尝试抓取
                initial_z_str = f"{self.initial_block_z:.3f}" if self.initial_block_z is not None else "None"
                logger.warning(f"  [Controller] ⚠️  抓取失败！block未被提升或未跟随 "
                             f"(block_z={block_pos[2]:.3f}, initial_z={initial_z_str}, "
                             f"ee_block_dist={dist_ee_block:.3f}m, block_lifted={block_lifted}, "
                             f"following={block_following})，重新尝试抓取")
                # 如果block掉到地面上，需要先打开gripper，然后重新approach
                if block_pos[2] < 0.05:
                    logger.info(f"  [Controller] Block在地面上，先打开gripper，然后重新approach")
                    self.phase = "approach_block"  # 重新从approach开始
                    # 更新initial_block_z为当前值（地面上的位置）
                    self.initial_block_z = block_pos[2]
                    action[3] = 0.0  # 打开gripper (0.0=打开, 2.0=关闭)
                else:
                    self.phase = "grasp"  # 如果block还在空中，直接从grasp重试
                self.grasp_step = 0
                # 重置lift相关变量
                self._lift_log_step = 0
                self._lift_start_ee_z = None
                self._lift_start_block_z = None
                self._grasp_close_wait_start = None
                
        elif self.phase == "move_to_target":
            # 阶段4: 移动到target位置正上方（保持gripper关闭）
            # 关键：必须精确对齐XY，然后再垂直插入，避免碰撞
            # 确保target_pos已初始化
            if self.target_pos is None:
                logger.error("  [Controller] ❌ target_pos未初始化！")
                return np.zeros(4, dtype=np.float32)
            
            # 两阶段策略：
            # 1. 先Z轴移动到安全高度（target上方至少20cm，确保brick下端不会碰到第3层）
            # 2. 然后XY精确对齐到target正上方（<5mm精度）
            safe_z_height = self.target_pos[2] + 0.20  # target上方20cm（第3层顶部约0.314m，第4层中心0.362m，安全高度0.562m）
            
            # 计算误差
            error_xy = self.target_pos[:2] - ee_pos[:2]
            error_z = safe_z_height - ee_pos[2]
            dist_xy_to_target = np.linalg.norm(error_xy)
            dist_z_to_safe = abs(error_z)
            
            # 优先策略：先Z轴到安全高度，然后XY对齐
            # 注意：EEActionWrapper的step_size是0.025m（2.5cm），所以action值[-1,1]对应[-2.5cm, 2.5cm]的移动
            wrapper_step_size = 0.025  # EEActionWrapper的默认step_size
            
            # 如果XY已经对齐（<5mm），即使Z还没到安全高度，也继续提升Z以便尽快进入place阶段
            # 但如果Z距离安全高度太远（>15cm），说明可能达到了工作空间限制，直接进入place阶段
            if dist_xy_to_target < 0.005:
                if dist_z_to_safe > 0.15:  # Z距离安全高度太远（>15cm），可能达到工作空间限制，直接进入place
                    logger.warning(f"  [Controller] ⚠️  XY已对齐但Z距离安全高度太远 ({dist_z_to_safe*1000:.1f}mm > 150mm)，直接进入place阶段")
                    self.phase = "place"
                    self.place_step = 0
                    action[:3] = [0, 0, 0]  # 保持位置，等待下一帧
                elif dist_z_to_safe > 0.02:  # XY已对齐但Z还没到安全高度，继续提升Z
                    # 只移动Z方向，保持XY不变
                    z_step_size = 0.025  # 2.5cm步长（与wrapper的step_size一致）
                    delta_z = np.clip(error_z, -z_step_size, z_step_size)
                    action[:3] = np.array([0, 0, delta_z / z_step_size])
            elif dist_z_to_safe > 0.10:  # Z高度还差很多（>10cm），优先提升Z
                # 只移动Z方向，保持XY不变
                # 使用较大的步长快速提升到安全高度
                z_step_size = 0.025  # 2.5cm步长（与wrapper的step_size一致）
                delta_z = np.clip(error_z, -z_step_size, z_step_size)
                action[:3] = np.array([0, 0, delta_z / z_step_size])
            elif dist_xy_to_target > 0.005:  # XY未对齐（>5mm），开始移动XY（Z已经接近安全高度）
                # 移动XY方向，同时如果Z还没到安全高度，也稍微提升Z
                # 根据XY距离自适应步长：距离远用大步长，距离近用小步长
                if dist_xy_to_target > 0.15:  # 距离>15cm，用大步长快速接近
                    xy_step_size = 0.025  # 2.5cm步长（与wrapper一致，最大值）
                elif dist_xy_to_target > 0.1:  # 距离10-15cm
                    xy_step_size = 0.02  # 2cm步长
                elif dist_xy_to_target > 0.05:  # 距离5-10cm，中等步长
                    xy_step_size = 0.015  # 1.5cm步长
                elif dist_xy_to_target > 0.02:  # 距离2-5cm，小步长
                    xy_step_size = 0.01  # 1cm步长
                else:  # 距离<2cm，精细步长
                    xy_step_size = 0.005  # 5mm步长
                
                delta_xy = error_xy.copy()
                # Clip到步长范围
                delta_xy = np.clip(delta_xy, -xy_step_size, xy_step_size)
                
                # 计算XY action：直接除以wrapper_step_size，这样wrapper会乘以0.025m得到实际移动距离
                # 例如：如果delta_xy = [0.025, 0.025]，action_xy = [1.0, 1.0]，wrapper会移动[2.5cm, 2.5cm]
                action_xy = delta_xy / wrapper_step_size
                
                # 同时检查Z：如果Z还没到安全高度，也稍微提升（但不能影响XY移动）
                if dist_z_to_safe > 0.02:  # Z还没到安全高度，同时提升Z（但优先级较低）
                    z_step_size = 0.015  # 较小的Z步长，优先保证XY移动
                    delta_z = np.clip(error_z, -z_step_size, z_step_size)
                    action_z = delta_z / wrapper_step_size
                else:
                    action_z = 0.0  # Z已在安全高度，不移动
                
                action[:3] = np.array([action_xy[0], action_xy[1], action_z])
                # 裁剪到[-1, 1]范围
                action[:3] = np.clip(action[:3], -1.0, 1.0)
            else:
                # XY已对齐（<5mm），Z也在安全高度，可以进入place阶段
                action[:3] = [0, 0, 0]  # 保持位置
            
            action[3] = 2.0  # 保持关闭，不要松开！(范围[0,2]: 0=打开, 2=关闭)
            
            # 调试输出（每5步打印一次，更频繁）
            if not hasattr(self, '_move_log_step'):
                self._move_log_step = 0
            self._move_log_step += 1
            if self._move_log_step % 5 == 0:  # 改为每5步打印一次
                # 确定当前策略
                if dist_xy_to_target < 0.005 and dist_z_to_safe > 0.15:
                    strategy = "XY已对齐，Z距离太远，直接进入place"
                    action_desc = "保持位置"
                elif dist_xy_to_target < 0.005 and dist_z_to_safe > 0.02:
                    strategy = "XY已对齐，继续提升Z到安全高度"
                    action_desc = f"Z={action[2]:.2f}"
                elif dist_z_to_safe > 0.10:
                    strategy = "提升Z到安全高度"
                    action_desc = f"Z={action[2]:.2f}"
                elif dist_xy_to_target > 0.005:
                    strategy = "对齐XY" + (f" (同时提升Z={action[2]:.2f})" if dist_z_to_safe > 0.02 else "")
                    action_desc = f"XY=[{action[0]:.2f}, {action[1]:.2f}]"
                else:
                    strategy = "等待进入place"
                    action_desc = "保持位置"
                logger.info(f"  [Controller] {self.phase}: target={self.target_pos}, ee={ee_pos}, "
                          f"dist_xy_to_target={dist_xy_to_target*1000:.1f}mm, dist_z_to_safe={dist_z_to_safe*1000:.1f}mm, "
                          f"safe_z={safe_z_height:.3f}m, 策略={strategy}, action={action_desc}")
            
            # 进入place阶段的条件：XY精确对齐（<5mm）且Z在安全高度（或Z距离安全高度<10cm，允许一定偏差）
            # 放宽条件：如果XY对齐且Z距离安全高度<10cm，就可以进入place（避免因Z无法提升而卡住）
            if dist_xy_to_target < 0.005 and dist_z_to_safe < 0.10:  # 放宽Z条件到10cm
                self.phase = "place"
                self.place_step = 0
                logger.info(f"  [Controller] ✅ 阶段转换: move_to_target -> place "
                          f"(dist_xy={dist_xy_to_target*1000:.1f}mm, Z高度安全)")
                
        elif self.phase == "place":
            # 阶段5: 垂直下降到target位置（保持gripper关闭，直到到达目标位置）
            # 多阶段策略：根据距离采用不同的控制策略
            # 1. 距离>10mm：正常下降，允许小的XY微调
            # 2. 距离5-10mm：精细对齐XY（停止Z下降），确保XY<2mm
            # 3. 距离<5mm：继续Z下降插入，同时保持XY对齐
            # 确保target_pos已初始化
            if self.target_pos is None:
                logger.error("  [Controller] ❌ target_pos未初始化！")
                return np.zeros(4, dtype=np.float32)
            
            wrapper_step_size = 0.025  # EEActionWrapper的默认step_size
            
            # 计算block到target的距离（关键指标）
            dist_block_target = np.linalg.norm(block_pos - self.target_pos)
            dist_block_target_xy = np.linalg.norm((block_pos - self.target_pos)[:2])
            dist_block_target_z = abs(block_pos[2] - self.target_pos[2])
            
            # 检查EE的XY对齐情况（用于控制）
            error_xy = self.target_pos[:2] - ee_pos[:2]
            error_xy_norm = np.linalg.norm(error_xy)
            
            # 如果XY偏差>15mm，返回move_to_target重新对齐
            if error_xy_norm > 0.015:  # XY偏差>15mm，必须重新对齐
                logger.warning(f"  [Controller] ⚠️  Place阶段XY偏差过大 ({error_xy_norm*1000:.1f}mm > 15mm)，返回move_to_target重新对齐")
                self.phase = "move_to_target"
                self.place_step = 0
                # 清除碰撞检测状态
                if hasattr(self, '_last_block_target_dist'):
                    delattr(self, '_last_block_target_dist')
                return self.get_action()  # 递归调用，重新计算动作
            
            # 多阶段策略：根据block到target的距离决定控制策略
            target_at_slot = self.target_pos.copy()
            error_z = target_at_slot[2] - ee_pos[2]
            
            # 初始化变量（确保所有分支都定义）
            action_xy_correction = np.array([0.0, 0.0])
            action_z = 0.0
            
            # 阶段1：距离>10mm，正常下降，允许小的XY微调
            if dist_block_target > 0.010:  # 距离>10mm
                # 自适应Z步长
                if abs(error_z) > 0.05:  # 距离 > 5cm
                    z_step_size = 0.015  # 1.5cm步长
                elif abs(error_z) > 0.02:  # 距离 2-5cm
                    z_step_size = 0.01  # 1cm步长
                else:  # 距离 < 2cm
                    z_step_size = 0.005  # 5mm步长
                
                # 允许小的XY微调（优先级低于Z下降）
                if error_xy_norm > 0.008:  # XY偏差>8mm
                    xy_correction = 0.3  # XY修正系数（较小）
                    error_xy_normalized = error_xy / (error_xy_norm + 1e-6)
                    xy_adjustment = error_xy_normalized * min(error_xy_norm, 0.005) * xy_correction
                    action_xy_correction = xy_adjustment / wrapper_step_size
                
                delta_z = np.clip(error_z, -z_step_size, z_step_size)
                action_z = delta_z / z_step_size
                
            # 阶段2：距离5-10mm，精细对齐XY（停止Z下降或非常慢）
            elif dist_block_target > 0.005:  # 距离5-10mm
                # 优先精细对齐XY，确保XY<2mm后再继续Z下降
                if dist_block_target_xy > 0.002:  # XY偏差>2mm，先对齐XY
                    # 只移动XY，停止Z移动
                    xy_step_size = 0.003  # 3mm步长，精细调整
                    error_xy_normalized = error_xy / (error_xy_norm + 1e-6)
                    delta_xy = np.clip(error_xy, -xy_step_size, xy_step_size)
                    action_xy_correction = delta_xy / wrapper_step_size
                    action_z = 0.0  # 停止Z下降，先对齐XY
                else:  # XY已对齐（<2mm），可以继续Z下降
                    # 使用较小的Z步长
                    z_step_size = 0.003  # 3mm步长，精细下降
                    delta_z = np.clip(error_z, -z_step_size, z_step_size)
                    action_z = delta_z / z_step_size
                    # action_xy_correction保持为[0, 0]（已在初始化时设置）
                
            # 阶段3：距离<5mm，最终精细插入
            else:  # 距离<5mm，接近目标，精细控制
                # 使用很小的步长，同时微调XY和Z
                z_step_size = 0.002  # 2mm步长，非常精细
                xy_step_size = 0.002  # 2mm步长，精细XY微调
                
                # XY微调
                if error_xy_norm > 0.001:  # XY偏差>1mm，继续微调
                    error_xy_normalized = error_xy / (error_xy_norm + 1e-6)
                    delta_xy = np.clip(error_xy, -xy_step_size, xy_step_size)
                    action_xy_correction = delta_xy / wrapper_step_size
                
                # Z下降（非常精细）
                delta_z = np.clip(error_z, -z_step_size, z_step_size)
                action_z = delta_z / z_step_size
            
            # 初始化碰撞检测状态
            if not hasattr(self, '_last_block_target_dist'):
                self._last_block_target_dist = dist_block_target
            
            # 碰撞检测：如果block到target的距离在下降过程中增加，说明可能发生碰撞
            if self.place_step > 5:
                # 如果block距离target增加了>8mm，可能发生碰撞，停止并尝试恢复
                if dist_block_target > self._last_block_target_dist + 0.008:
                    logger.warning(f"  [Controller] ⚠️  检测到可能的碰撞！block距离target增加 "
                                 f"({self._last_block_target_dist*1000:.1f}mm -> {dist_block_target*1000:.1f}mm)")
                    # 稍微上移，避免继续碰撞
                    action[:3] = [0, 0, 0.3]  # 轻微上移
                    self._last_block_target_dist = dist_block_target
                    action[3] = 2.0  # 保持关闭
                    return action.astype(np.float32)
                else:
                    self._last_block_target_dist = dist_block_target
            
            # 组合动作：根据阶段策略组合XY和Z动作
            action[:3] = np.array([action_xy_correction[0], action_xy_correction[1], action_z])
            action[:3] = np.clip(action[:3], -1.0, 1.0)
            action[3] = 2.0  # 保持关闭，不要松开！(范围[0,2]: 0=打开, 2=关闭)
            
            self.place_step += 1
            # 计算到目标的精确距离（已在上方计算）
            dist_ee_target_z = abs(error_z)
            
            # 确定当前阶段（用于日志）
            if dist_block_target > 0.010:
                place_sub_phase = "正常下降"
            elif dist_block_target > 0.005:
                place_sub_phase = "精细对齐XY" if dist_block_target_xy > 0.002 else "继续下降"
            else:
                place_sub_phase = "最终精细插入"
            
            # 调试输出
            if self.place_step % 5 == 0:
                logger.info(f"  [Controller] {self.phase} ({place_sub_phase}, step={self.place_step}): "
                          f"target={target_at_slot}, ee={ee_pos}, block={block_pos}, "
                          f"block_target_3d={dist_block_target*1000:.1f}mm, "
                          f"block_target_xy={dist_block_target_xy*1000:.1f}mm, "
                          f"block_target_z={dist_block_target_z*1000:.1f}mm, "
                          f"ee_target_xy={error_xy_norm*1000:.1f}mm, action_xy=[{action[0]:.2f}, {action[1]:.2f}], "
                          f"action_z={action[2]:.2f}, gripper=关闭")
            
            # 改进的释放条件：block必须非常接近目标（严格匹配环境的阈值）
            # 环境阈值：XY<5mm, Z<3mm, 3D<5mm
            # 控制器使用更严格的条件（略小于阈值），确保满足环境要求
            xy_ok = dist_block_target_xy < 0.004  # 4mm（略小于5mm阈值，留有余量）
            z_ok = dist_block_target_z < 0.0025   # 2.5mm（略小于3mm阈值，留有余量）
            distance_ok = dist_block_target < 0.0045  # 4.5mm（略小于5mm阈值，留有余量）
            
            if xy_ok and z_ok and distance_ok:
                # 到达目标位置，可以释放了
                self.phase = "release"
                self.release_step = 0
                logger.info(f"  [Controller] ✅ 阶段转换: place -> release "
                          f"(place_step={self.place_step}, block_target_3d={dist_block_target*1000:.1f}mm, "
                          f"xy={dist_block_target_xy*1000:.1f}mm, z={dist_block_target_z*1000:.1f}mm)")
            elif self.place_step > 60:
                # 超时（60步=6秒），如果已经很接近（距离<2cm），也尝试释放
                if dist_block_target < 0.02:
                    logger.warning(f"  [Controller] ⚠️  放置超时但已接近目标，尝试释放 "
                                 f"(place_step={self.place_step}, block_target={dist_block_target*1000:.1f}mm)")
                    self.phase = "release"
                    self.release_step = 0
                else:
                    logger.warning(f"  [Controller] ⚠️  放置超时且距离较远，强制释放 "
                                 f"(place_step={self.place_step}, block_target={dist_block_target*1000:.1f}mm)")
                    self.phase = "release"
                    self.release_step = 0
                
        elif self.phase == "release":
            # 阶段6: 打开gripper并稍微上移
            # 关键：必须确保gripper真正打开（joint角度>0.4）才能判定为成功释放
            # 延长release阶段从20步到40步，给gripper更多时间完全打开
            self.release_step += 1
            
            # 获取当前gripper状态
            right_angle, left_angle = self.get_gripper_joint_angles()
            is_gripper_open = False
            if right_angle is not None and left_angle is not None:
                is_gripper_open = (right_angle > 0.4) and (left_angle > 0.4)  # 阈值从0.5放宽到0.4
            
            # 计算TCP到block的距离（用于验证释放）
            dist_tcp_block = np.linalg.norm(block_pos - ee_pos) if block_pos is not None and ee_pos is not None else None
            
            if self.release_step <= 40:
                # 前40步：保持位置，持续发送打开命令（gripper控制是累加的，需要持续发送）
                # 延长保持位置时间，确保gripper有足够时间完全打开
                action[:3] = [0, 0, 0]
                action[3] = 0.0  # gripper打开 (范围[0,2]: 0=打开, 2=关闭)
                if self.release_step == 1:
                    logger.info(f"  [Controller] {self.phase}: 开始打开gripper释放砖块 (step={self.release_step})")
                elif self.release_step % 5 == 0:
                    angle_str = f"R={right_angle:.3f}, L={left_angle:.3f}" if right_angle is not None else "unknown"
                    dist_str = f"{dist_tcp_block*1000:.1f}mm" if dist_tcp_block is not None else "unknown"
                    logger.info(f"  [Controller] {self.phase} (step={self.release_step}): "
                              f"gripper打开中... angles={angle_str}, TCP_dist={dist_str}, "
                              f"is_open={is_gripper_open}")
            elif self.release_step <= 50:
                # 第41-50步：稍微上移（1cm），继续打开gripper
                action[:3] = [0, 0, 0.1]  # 上移1cm
                action[3] = 0.0  # 继续打开gripper (0.0=打开, 2.0=关闭)
                if self.release_step == 41:
                    logger.info(f"  [Controller] {self.phase}: 上移gripper远离砖块 (step={self.release_step})")
                elif self.release_step % 5 == 0:
                    angle_str = f"R={right_angle:.3f}, L={left_angle:.3f}" if right_angle is not None else "unknown"
                    dist_str = f"{dist_tcp_block*1000:.1f}mm" if dist_tcp_block is not None else "unknown"
                    logger.info(f"  [Controller] {self.phase} (step={self.release_step}): "
                              f"上移中... angles={angle_str}, TCP_dist={dist_str}, is_open={is_gripper_open}")
            else:
                # 第51步之后：继续上移（更远），确保完全分离
                action[:3] = [0, 0, 0.2]  # 继续上移
                action[3] = 0.0  # 保持打开 (0.0=打开, 2.0=关闭)
                if self.release_step == 51:
                    logger.info(f"  [Controller] {self.phase}: 继续上移确保完全分离 (step={self.release_step})")
                    if is_gripper_open:
                        logger.info(f"  [Controller] ✅ Gripper已成功打开 (angles: R={right_angle:.3f}, L={left_angle:.3f})")
                    else:
                        logger.warning(f"  [Controller] ⚠️  Gripper可能未完全打开 (angles: R={right_angle:.3f}, L={left_angle:.3f})")
        
        return action.astype(np.float32)


def verify_video_file(video_path: Path, max_attempts: int = 3) -> bool:
    """
    验证视频文件是否完整且可解码
    
    Args:
        video_path: 视频文件路径
        max_attempts: 最大尝试次数（用于处理文件系统同步延迟）
    
    Returns:
        True if video is valid, False otherwise
    """
    for attempt in range(max_attempts):
        if not video_path.exists():
            if attempt < max_attempts - 1:
                time.sleep(0.2)  # 等待文件系统同步
                continue
            logger.warning(f"视频文件不存在: {video_path}")
            return False
        
        try:
            # 尝试打开并读取视频文件
            with av.open(str(video_path), "r") as container:
                # 检查是否有视频流
                if len(container.streams.video) == 0:
                    logger.warning(f"视频文件没有视频流: {video_path}")
                    return False
                
                video_stream = container.streams.video[0]
                # 尝试解码第一帧
                frame_count = 0
                for frame in container.decode(video_stream):
                    frame_count += 1
                    if frame_count >= 1:  # 至少解码一帧即可
                        break
                
                if frame_count == 0:
                    logger.warning(f"无法从视频文件解码任何帧: {video_path}")
                    return False
                
                # 验证文件大小（应该大于0）
                file_size = video_path.stat().st_size
                if file_size == 0:
                    logger.warning(f"视频文件大小为0: {video_path}")
                    return False
                
                logger.debug(f"✅ 视频文件验证成功: {video_path} (大小: {file_size} bytes, 帧数: {frame_count})")
                return True
                
        except Exception as e:
            # PyAV 可能抛出各种异常（AVError, OSError等），统一处理
            if attempt < max_attempts - 1:
                logger.warning(f"视频文件验证失败（尝试 {attempt + 1}/{max_attempts}）: {e}，等待后重试...")
                time.sleep(0.3)
                continue
            logger.error(f"视频文件验证失败: {video_path}, 错误: {type(e).__name__}: {e}")
            return False
    
    return False


def verify_episode_videos(dataset, episode_index: int) -> bool:
    """
    验证episode的所有视频文件是否完整
    
    Args:
        dataset: LeRobotDataset实例
        episode_index: episode索引
    
    Returns:
        True if all videos are valid, False otherwise
    """
    if not hasattr(dataset, 'meta') or dataset.meta.episodes is None:
        return True  # 无法验证，假设有效
    
    if episode_index >= len(dataset.meta.episodes):
        logger.warning(f"Episode {episode_index} 的元数据不存在")
        return False
    
    episode_meta = dataset.meta.episodes[episode_index]
    all_valid = True
    
    # 检查所有视频键
    for video_key in dataset.meta.video_keys:
        try:
            chunk_idx = episode_meta[f"videos/{video_key}/chunk_index"]
            file_idx = episode_meta[f"videos/{video_key}/file_index"]
            video_path = dataset.root / dataset.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            
            if not verify_video_file(video_path):
                logger.error(f"❌ Episode {episode_index} 的视频文件损坏: {video_path}")
                all_valid = False
            else:
                logger.debug(f"✅ Episode {episode_index} 的视频文件有效: {video_key}")
        except KeyError as e:
            logger.warning(f"Episode {episode_index} 缺少视频元数据键: {e}")
            all_valid = False
        except Exception as e:
            logger.error(f"验证episode {episode_index} 视频时发生错误: {e}")
            all_valid = False
    
    return all_valid


def auto_collect_dataset(
    config_path: str,
    num_episodes: int = 10,
    output_dir: str = None,
    enable_camera_display: bool = False,  # 默认禁用，避免影响数据采集
    camera_display_freq: int = 5,  # 如果启用，每N帧显示一次（降低频率）
):
    """
    自动化采集数据集
    
    Args:
        config_path: 配置文件路径
        num_episodes: 要采集的episode数量
        output_dir: 输出目录（如果不指定，会自动生成）
        fps: 帧率
    """
    from lerobot.rl.acfql.gym_manipulator import GymManipulatorConfig
    from lerobot.rl.gym_manipulator import DatasetConfig
    import json
    import draccus
    
    # register_third_party_devices已经在模块导入时调用了
    
    # 使用draccus解析配置文件
    # 临时修改argv以避免参数冲突
    import sys
    original_argv = sys.argv
    try:
        # 临时设置argv，只包含config_path，这样draccus只会从文件加载，不会解析命令行参数
        sys.argv = ['auto_collect_masonry_data.py', f'--config_path={config_path}']
        # 使用draccus从文件加载配置，args=[]表示不处理额外的命令行参数
        cfg = draccus.parse(config_class=GymManipulatorConfig, config_path=config_path, args=[])
    finally:
        sys.argv = original_argv
    
    # 读取JSON以便后续使用
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 设置mode为record
    cfg.mode = "record"
    
    # 设置数据集参数
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(__file__).parent.parent / "datasets" / "masonry_insertion_acfql"
        output_dir = f"{base_dir}_{timestamp}"
    
    # 确保dataset配置存在
    if not hasattr(cfg, 'dataset') or cfg.dataset is None:
        from lerobot.rl.gym_manipulator import DatasetConfig
        from omegaconf import OmegaConf
        cfg.dataset = OmegaConf.structured(DatasetConfig(
            repo_id=config_dict.get('dataset', {}).get('repo_id', 'masonry_insertion_acfql'),
            root=output_dir,
            task=config_dict.get('dataset', {}).get('task', 'MasonryBlockInsertionGamepad-v0'),
            num_episodes_to_record=num_episodes,
            push_to_hub=False,
        ))
    else:
        cfg.dataset.root = output_dir
        cfg.dataset.num_episodes_to_record = num_episodes
        cfg.dataset.push_to_hub = False
    
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"将采集 {num_episodes} 个episodes")
    
    # 创建Base环境（没有InputsControlWrapper），然后手动应用需要的wrapper
    # 这样可以完全控制action传递，不受gamepad干预
    import gym_hil  # noqa: F401
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True
    gripper_penalty = cfg.env.processor.gripper.gripper_penalty if cfg.env.processor.gripper is not None else 0.0
    
    # 使用Base环境ID，避免InputsControlWrapper
    base_task = cfg.env.task.replace("Gamepad", "Base").replace("Keyboard", "Base").replace("MetaQuest", "Base")
    if base_task != cfg.env.task:
        logger.info(f"自动将task从 {cfg.env.task} 改为 {base_task}，以避免InputsControlWrapper")
    
    # 创建Base环境（直接创建，不通过make_robot_env，避免InputsControlWrapper）
    # Base环境不接受use_gripper和gripper_penalty参数，这些在wrapper中处理
    base_env = gym.make(
        f"gym_hil/{base_task}",
        image_obs=True,
        render_mode="human",
    )
    
    # 手动应用必要的wrapper（与factory.py中的wrap_env一致）
    # 1. GripperPenaltyWrapper
    if use_gripper:
        base_env = GripperPenaltyWrapper(base_env, penalty=gripper_penalty)
    
    # 2. EEActionWrapper（关键！将[delta_x, delta_y, delta_z, gripper]转换为7D格式）
    ee_step_size = DEFAULT_EE_STEP_SIZE
    base_env = EEActionWrapper(
        base_env, 
        ee_action_step_size=ee_step_size, 
        use_gripper=True, 
        use_6dof=False  # 3-DoF模式
    )
    
    # 3. PassiveViewerWrapper (如果需要可视化)
    base_env = PassiveViewerWrapper(base_env, show_left_ui=True, show_right_ui=True)
    
    # 4. ResetDelayWrapper
    reset_delay = cfg.env.processor.reset.reset_time_s if cfg.env.processor.reset is not None else 1.0
    base_env = ResetDelayWrapper(base_env, delay_seconds=reset_delay)
    
    # 5. 设置terminate_on_success参数（从配置中读取）
    terminate_on_success = cfg.env.processor.reset.terminate_on_success if cfg.env.processor.reset is not None else True
    # 获取底层环境（可能需要unwrap多层wrapper）
    unwrapped_env = base_env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    # 设置terminate_on_success属性
    if hasattr(unwrapped_env, '_terminate_on_success'):
        unwrapped_env._terminate_on_success = terminate_on_success
        logger.info(f"设置 terminate_on_success = {terminate_on_success}")
    else:
        logger.warning(f"⚠️  环境不支持 terminate_on_success 参数")
    
    # 使用我们创建的环境（没有InputsControlWrapper）
    env = base_env
    env_processor, action_processor = make_processors(env, None, cfg.env, cfg.device)
    
    # 创建自动控制器
    controller = AutomaticMasonryController(env)
    
    # 获取action维度
    action_dim = env.action_space.shape[0]
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper else False
    
    # 使用与gym_manipulator相同的数据集创建方式
    # 获取初始observation以确定特征
    obs, info = env.reset()
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)
    
    # 构建features字典（与gym_manipulator一致）
    action_features = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": None,
    }
    features = {
        ACTION: action_features,
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
    }
    if use_gripper:
        features["complementary_info.discrete_penalty"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        }
    
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
    
    for key, value in transition[TransitionKey.OBSERVATION].items():
        if key == OBS_STATE:
            features[key] = {
                "dtype": "float32",
                "shape": value.squeeze(0).shape,
                "names": None,
            }
        elif "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": value.squeeze(0).shape,
                "names": ["channels", "height", "width"],
            }
        else:
            # 处理其他observation keys
            val_shape = value.squeeze(0).shape if isinstance(value, torch.Tensor) else np.array(value).shape
            features[key] = {
                "dtype": "float32",
                "shape": val_shape,
                "names": None,
            }
    
    # 创建数据集
    # 关键设置：
    # - batch_encoding_size=1: 每个episode后立即编码视频，避免批处理导致的不完整
    # - image_writer_threads=4: 图像写入线程数
    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.env.fps,
        root=cfg.dataset.root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        batch_encoding_size=1,  # 立即编码，避免批处理导致视频不完整
        features=features,
    )
    
    # 控制循环（使用配置文件中的fps）
    dt = 1.0 / cfg.env.fps
    episode_idx = 0
    
    # 初始化相机可视化（可选，默认禁用以避免影响数据采集）
    display_camera_views = enable_camera_display and isinstance(obs, dict) and "pixels" in obs
    camera_display_counter = 0  # 用于控制显示频率
    if display_camera_views:
        import cv2
        cv2.namedWindow("front", cv2.WINDOW_NORMAL)
        cv2.namedWindow("wrist", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("front", 256, 256)
        cv2.resizeWindow("wrist", 256, 256)
        logger.info(f"📹 Camera views initialized: 'front' and 'wrist' windows (显示频率: 每{camera_display_freq}帧)")
    else:
        logger.info("📹 Camera display disabled (recommended for stable data collection)")
    
    while episode_idx < num_episodes:
        # Reset环境
        obs, info = env.reset()
        env_processor.reset()
        action_processor.reset()
        controller.reset()  # 使用reset方法重置控制器
        camera_display_counter = 0  # 重置相机显示计数器
        
        # 创建初始transition
        transition = create_transition(observation=obs, info=info)
        transition = env_processor(transition)
        
        episode_start_time = time.perf_counter()
        episode_step = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始采集 Episode {episode_idx + 1}/{num_episodes}")
        logger.info(f"{'='*60}")
        
        while True:
            step_start_time = time.perf_counter()
            
            # 从控制器获取动作
            action = controller.get_action()
            
            # 转换为tensor
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            
            # 执行动作
            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            
            # 获取当前observation（用于相机显示）
            obs = transition[TransitionKey.OBSERVATION]
            
            terminated = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)
            
            # 记录数据 - 与gym_manipulator完全一致的格式
            observations = {
                k: v.squeeze(0).cpu() if isinstance(v, torch.Tensor) else v
                for k, v in transition[TransitionKey.OBSERVATION].items()
            }
            
            action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                "teleop_action", transition[TransitionKey.ACTION]
            )
            if isinstance(action_to_record, torch.Tensor):
                action_to_record = action_to_record.squeeze(0).cpu()
            else:
                action_to_record = torch.tensor(action_to_record).squeeze(0).cpu() if hasattr(action_to_record, '__len__') else action_to_record
            
            frame = {
                **observations,
                ACTION: action_to_record,
                REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                DONE: np.array([terminated], dtype=bool),
                "task": cfg.dataset.task,
            }
            
            if use_gripper:
                discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)
                frame["complementary_info.discrete_penalty"] = np.array([discrete_penalty], dtype=np.float32)
            
            dataset.add_frame(frame)
            
            episode_step += 1
            
            # 显示相机视图（降低频率以避免影响数据采集）
            if display_camera_views:
                camera_display_counter += 1
                # 只每N帧显示一次，降低对主循环的影响
                if camera_display_counter >= camera_display_freq:
                    camera_display_counter = 0
                    import cv2
                    # 从processed observation获取图像
                    front_img = transition[TransitionKey.OBSERVATION].get("observation.images.front")
                    wrist_img = transition[TransitionKey.OBSERVATION].get("observation.images.wrist")
                    
                    # 如果processed observation中没有，尝试从原始observation获取
                    if front_img is None or wrist_img is None:
                        # 尝试从processed observation中的其他键获取
                        if isinstance(obs, dict):
                            # 检查是否有其他格式的图像键
                            for key in ["observation.images.front", "pixels"]:
                                if key in obs:
                                    if isinstance(obs[key], dict):
                                        front_img = obs[key].get("front", front_img)
                                        wrist_img = obs[key].get("wrist", wrist_img)
                                    break
                    
                    # 显示front相机视图
                    if front_img is not None:
                        # 转换为numpy array
                        if isinstance(front_img, torch.Tensor):
                            front_img = front_img.squeeze(0).cpu().numpy()
                        else:
                            front_img = np.asarray(front_img)
                        
                        # 转换格式: (C, H, W) -> (H, W, C)
                        if len(front_img.shape) == 3 and front_img.shape[0] == 3:
                            front_img = np.transpose(front_img, (1, 2, 0))
                        
                        # 确保值在[0, 255]范围内
                        if front_img.max() <= 1.0:
                            front_img = (front_img * 255).astype(np.uint8)
                        
                        # RGB -> BGR for OpenCV
                        front_img_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("front", front_img_bgr)
                    
                    # 显示wrist相机视图
                    if wrist_img is not None:
                        # 转换为numpy array
                        if isinstance(wrist_img, torch.Tensor):
                            wrist_img = wrist_img.squeeze(0).cpu().numpy()
                        else:
                            wrist_img = np.asarray(wrist_img)
                        
                        # 转换格式: (C, H, W) -> (H, W, C)
                        if len(wrist_img.shape) == 3 and wrist_img.shape[0] == 3:
                            wrist_img = np.transpose(wrist_img, (1, 2, 0))
                        
                        # 确保值在[0, 255]范围内
                        if wrist_img.max() <= 1.0:
                            wrist_img = (wrist_img * 255).astype(np.uint8)
                        
                        # RGB -> BGR for OpenCV
                        wrist_img_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("wrist", wrist_img_bgr)
                    
                    cv2.waitKey(1)
            
            # 检查episode结束
            if terminated or truncated:
                episode_time = time.perf_counter() - episode_start_time
                success = transition[TransitionKey.INFO].get("succeed", False)
                logger.info(
                    f"Episode {episode_idx + 1} 结束: {episode_step} 步, "
                    f"{episode_time:.1f}秒, 成功: {success}, "
                    f"奖励: {transition[TransitionKey.REWARD]:.4f}"
                )
                
                # 保存episode（关键：必须成功，否则数据不完整）
                # save_episode()会等待图像写入和视频编码完成，所以是同步的
                try:
                    logger.info(f"正在保存 Episode {episode_idx + 1}...")
                    dataset.save_episode()
                    # 额外等待，确保文件系统完全同步（视频编码和文件写入可能需要时间）
                    # 增加等待时间到1秒，确保视频文件完全写入磁盘
                    time.sleep(1.0)
                    
                    # 验证episode是否成功保存（检查元数据）
                    if hasattr(dataset, 'meta') and dataset.meta.episodes is not None:
                        if len(dataset.meta.episodes) > episode_idx:
                            logger.info(f"✅ Episode {episode_idx + 1} 元数据已保存")
                        else:
                            logger.warning(f"⚠️  Episode {episode_idx + 1} 元数据可能未完全写入")
                            time.sleep(0.5)  # 额外等待
                            continue  # 跳过验证，等待下一帧
                    else:
                        logger.info(f"✅ Episode {episode_idx + 1} 保存成功（元数据不可用）")
                    
                    # 验证视频文件完整性（关键！）
                    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'video_keys') and len(dataset.meta.video_keys) > 0:
                        logger.info(f"正在验证 Episode {episode_idx + 1} 的视频文件...")
                        if verify_episode_videos(dataset, episode_idx):
                            logger.info(f"✅ Episode {episode_idx + 1} 保存成功（视频文件已验证）")
                        else:
                            logger.error(f"❌ Episode {episode_idx + 1} 的视频文件验证失败！")
                            # 视频文件损坏是严重错误，应该停止采集
                            raise RuntimeError(f"Episode {episode_idx + 1} 的视频文件损坏，停止采集以避免生成损坏的数据集")
                    else:
                        logger.info(f"✅ Episode {episode_idx + 1} 保存成功（无视频文件需要验证）")
                        
                except Exception as e:
                    logger.error(f"❌ Episode {episode_idx + 1} 保存失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 保存失败是严重错误，应该停止采集，避免生成损坏的数据集
                    logger.error("⚠️  保存失败，停止采集以避免生成损坏的数据集")
                    raise
                episode_idx += 1
                break
            
            # 超时检查（最多250步 = 25秒@10fps，与环境max_episode_steps一致）
            if episode_step >= 250:
                logger.warning(f"Episode {episode_idx + 1} 超时，强制结束")
                try:
                    logger.info(f"正在保存 Episode {episode_idx + 1}（超时）...")
                    dataset.save_episode()
                    # 额外等待，确保文件系统完全同步
                    time.sleep(1.0)
                    
                    # 验证episode是否成功保存
                    if hasattr(dataset, 'meta') and dataset.meta.episodes is not None:
                        if len(dataset.meta.episodes) > episode_idx:
                            logger.info(f"✅ Episode {episode_idx + 1} 元数据已保存（超时）")
                        else:
                            logger.warning(f"⚠️  Episode {episode_idx + 1} 元数据可能未完全写入（超时）")
                            time.sleep(0.5)  # 额外等待
                    
                    # 验证视频文件完整性（关键！）
                    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'video_keys') and len(dataset.meta.video_keys) > 0:
                        logger.info(f"正在验证 Episode {episode_idx + 1} 的视频文件（超时）...")
                        if verify_episode_videos(dataset, episode_idx):
                            logger.info(f"✅ Episode {episode_idx + 1} 保存成功（超时，视频文件已验证）")
                        else:
                            logger.error(f"❌ Episode {episode_idx + 1} 的视频文件验证失败（超时）！")
                            raise RuntimeError(f"Episode {episode_idx + 1} 的视频文件损坏（超时），停止采集以避免生成损坏的数据集")
                    else:
                        logger.info(f"✅ Episode {episode_idx + 1} 保存成功（超时，无视频文件需要验证）")
                        
                except Exception as e:
                    logger.error(f"❌ Episode {episode_idx + 1} 保存失败（超时）: {e}")
                    import traceback
                    traceback.print_exc()
                    # 保存失败是严重错误，应该停止采集
                    logger.error("⚠️  保存失败，停止采集以避免生成损坏的数据集")
                    raise
                episode_idx += 1
                break
            
            # 维持fps
            busy_wait(dt - (time.perf_counter() - step_start_time))
    
    # 关闭数据集（按照正确顺序）
    # 1. 先停止图像写入器，等待所有图像写入完成
    logger.info("停止图像写入器...")
    try:
        dataset.stop_image_writer()
        # 等待图像写入器完全停止
        time.sleep(0.5)
        logger.info("✅ 图像写入器已停止")
    except Exception as e:
        logger.error(f"❌ 停止图像写入器失败: {e}")
        raise
    
    # 2. 如果使用批处理编码（batch_encoding_size > 1），需要确保所有剩余的视频都被编码
    if hasattr(dataset, 'batch_encoding_size') and dataset.batch_encoding_size > 1:
        if hasattr(dataset, 'episodes_since_last_encoding') and dataset.episodes_since_last_encoding > 0:
            logger.info(f"编码剩余的 {dataset.episodes_since_last_encoding} 个episode的视频...")
            try:
                start_ep = dataset.num_episodes - dataset.episodes_since_last_encoding
                end_ep = dataset.num_episodes
                dataset._batch_save_episode_video(start_ep, end_ep)
                time.sleep(0.5)  # 等待批处理编码完成
                logger.info("✅ 剩余视频编码完成")
            except Exception as e:
                logger.error(f"❌ 批处理编码失败: {e}")
                raise
    
    # 3. 调用finalize()确保所有数据正确写入（关键！）
    # finalize()会：
    # - 刷新所有缓冲的episode元数据到磁盘
    # - 关闭parquet writers以写入footer元数据
    # - 聚合episode文件到chunk文件
    # - 确保数据集可以正确加载
    logger.info("完成数据集写入（finalize）...")
    try:
        dataset.finalize()
        # 等待finalize完成，确保所有文件完全写入
        time.sleep(0.5)
        logger.info("✅ 数据集finalize成功")
    except Exception as e:
        logger.error(f"❌ 数据集finalize失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    logger.info(f"数据集已保存到: {output_dir}")
    logger.info(f"共采集 {episode_idx} 个episodes")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="自动化采集masonry insertion数据集")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/simulation/acfql/masonry_insertion_gamepad.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="要采集的episode数量",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（可选，默认自动生成）",
    )
    parser.add_argument(
        "--enable_camera_display",
        action="store_true",
        help="启用相机可视化窗口（默认禁用，推荐禁用以确保数据采集稳定性）",
    )
    parser.add_argument(
        "--camera_display_freq",
        type=int,
        default=5,
        help="如果启用相机显示，每N帧显示一次（默认5，降低频率以减少对主循环的影响）",
    )
    args = parser.parse_args()
    
    output_dir = auto_collect_dataset(
        config_path=args.config,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        enable_camera_display=args.enable_camera_display,
        camera_display_freq=args.camera_display_freq,
    )
    
    # 关闭相机窗口
    try:
        import cv2
        cv2.destroyAllWindows()
        print("📹 相机窗口已关闭")
    except:
        pass
    
    print(f"\n✅ 数据集采集完成！")
    print(f"📁 数据保存在: {output_dir}")
    print(f"\n可以使用以下命令查看数据集:")
    print(f"  python -m lerobot.scripts.push_dataset_to_hub --data_dir {output_dir}")


if __name__ == "__main__":
    main()

