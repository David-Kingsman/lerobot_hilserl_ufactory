#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动化采集pick lift任务数据的脚本
使用基于位置的简单控制器自动执行pick和lift任务，替代人工操作
完全复用gym_manipulator的数据采集流程，只是用自动控制器替换gamepad输入
支持FT和非FT两种环境版本
# 采集非FT版本数据（默认）
python scripts/auto_collect_pick_lift_data.py \
    --config configs/simulation/acfql/gym_hil_env_fql.json \
    --num_episodes 50 

# 采集FT版本数据
python scripts/auto_collect_pick_lift_data.py \
    --config configs/simulation/acfql/gym_hil_env_fql_ft.json \
    --num_episodes 50 \
    --use_ft

# 启用相机显示（可选，默认禁用）
python scripts/auto_collect_pick_lift_data.py \
    --config configs/simulation/acfql/gym_hil_env_fql_ft.json \
    --num_episodes 50 \
    --use_ft \
    --enable_camera_display
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


class AutomaticPickLiftController:
    """自动控制器，用于pick lift任务
    模拟人类操作：pick cube -> lift -> place -> release
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
        
        self.phase = "approach_cube"  # 简化：只有3个阶段 approach_cube, grasp, lift
        self.grasp_step = 0
        self.initial_cube_z = None  # 记录初始block Z位置，用于判断是否成功举起
        self.env_z_init = None  # 环境的_z_init（reset时记录的初始block高度）
        self.success_detected = False  # 标记是否检测到success
        self.success_hold_steps = 0  # success后保持的步数（模拟人工采集行为）
        
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
        self.phase = "approach_cube"
        self.grasp_step = 0
        self.initial_cube_z = None  # 重置初始block Z位置
        self.env_z_init = None  # 重置环境的_z_init
        self.success_detected = False  # 重置success检测标志
        self.success_hold_steps = 0  # 重置success保持步数
        
        # 从环境获取_z_init（reset时记录的初始block高度）
        try:
            if hasattr(self.base_env, '_z_init'):
                self.env_z_init = self.base_env._z_init
                logger.debug(f"  [Controller] 获取环境的_z_init: {self.env_z_init:.3f}")
        except Exception as e:
            logger.warning(f"  [Controller] 无法获取环境的_z_init: {e}")
        
        # 重置日志计数器和grasp等待变量
        self._lift_log_step = 0
        if hasattr(self, '_grasp_close_wait_start'):
            delattr(self, '_grasp_close_wait_start')
        if hasattr(self, '_lift_hold_steps'):
            delattr(self, '_lift_hold_steps')
        logger.info(f"  [Controller] 🔄 重置控制器，初始阶段: {self.phase}")
        
    def get_action(self):
        """根据当前环境状态生成动作 [delta_x, delta_y, delta_z, gripper]
        简化逻辑：block位置固定，只需移动到上方 -> 下降抓取 -> 举起
        """
        cube_pos = None
        ee_pos = None
        
        try:
            # 从底层环境获取block位置（使用sensor）
            if hasattr(self.base_env, '_data'):
                try:
                    cube_pos = self.base_env._data.sensor("block_pos").data.copy()
                except Exception as e:
                    logger.debug(f"Failed to get cube_pos: {e}")
                
                # 获取end-effector位置
                try:
                    ee_pos = self.base_env._data.sensor("2f85/pinch_pos").data.copy()
                except Exception as e1:
                    try:
                        if hasattr(self.base_env, '_model'):
                            import mujoco
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
                    
                    if ee_pos is None and hasattr(self.base_env, '_ee_site_id') and self.base_env._ee_site_id is not None:
                        ee_pos = self.base_env._data.site_xpos[self.base_env._ee_site_id].copy()
        except Exception as e:
            if not hasattr(self, '_error_logged'):
                logger.warning(f"Error getting positions: {e}")
                self._error_logged = True
        
        # Block位置固定：[0.5, 0.0, 0.06]（pick lift环境的默认位置）
        if cube_pos is None:
            cube_pos = np.array([0.5, 0.0, 0.06])
        
        if ee_pos is None:
            ee_pos = np.array([0.5, 0.0, 0.3])
        
        # 计算delta动作 [delta_x, delta_y, delta_z, gripper]
        action = np.zeros(4, dtype=np.float32)
        action[3] = 0.0  # 关键修复：默认gripper为打开状态（0.0），确保reset后第一帧gripper是打开的
        step_size = 0.025  # 每步移动2.5cm
        
        # 简化逻辑：只有3个阶段
        # 关键修复：匹配人工数据的分布
        # - 增加episode长度（增加等待时间、探索）
        # - 确保gripper在大部分时间都是关闭的（匹配人工数据mean=1.92）
        # - 平滑动作变化（匹配人工数据std=0.33）
        if self.phase == "approach_cube":
            # 阶段1: 移动到block上方合适的高度（block位置固定：[0.5, 0.0, 0.06]）
            # 关键修复：如果初始高度已经足够高（>= block上方8cm），直接进入grasp阶段
            # 如果初始高度较低（< block上方8cm），先移动到block上方10cm，确保有足够的下降空间
            target_z = cube_pos[2] + 0.10  # block上方10cm（approach的目标高度）
            min_approach_height = cube_pos[2] + 0.08  # block上方8cm（最小approach高度）
            
            # 如果已经在足够高的高度（>= block上方8cm），直接进入grasp阶段，避免不必要的向上移动
            if ee_pos[2] >= min_approach_height:
                action[3] = 0.0  # 确保gripper保持打开
                if not hasattr(self, '_approach_wait_steps'):
                    self._approach_wait_steps = 0
                self._approach_wait_steps += 1
                if self._approach_wait_steps >= 1:  # 等待1步后进入grasp阶段
                    self.phase = "grasp"
                    self.grasp_step = 0
                    if hasattr(self, '_approach_wait_steps'):
                        delattr(self, '_approach_wait_steps')
                    logger.info(f"  [Controller] ✅ 已经在足够高度 (ee_z={ee_pos[2]:.3f} >= {min_approach_height:.3f})，直接进入抓取阶段")
            # 如果接近目标高度（block上方10cm ± 2cm），也直接进入grasp阶段
            elif abs(ee_pos[2] - target_z) < 0.02:
                action[3] = 0.0  # 确保gripper保持打开
                if not hasattr(self, '_approach_wait_steps'):
                    self._approach_wait_steps = 0
                self._approach_wait_steps += 1
                if self._approach_wait_steps >= 1:
                    self.phase = "grasp"
                    self.grasp_step = 0
                    if hasattr(self, '_approach_wait_steps'):
                        delattr(self, '_approach_wait_steps')
                    logger.info(f"  [Controller] ✅ 到达block上方 (ee_z={ee_pos[2]:.3f}, target_z={target_z:.3f})，进入抓取阶段")
            else:
                # 需要移动到block上方10cm（只在初始高度较低时才需要）
                delta_z = target_z - ee_pos[2]
                # 修复：移除multiplier限制，让delta_z能够达到完整的归一化范围[-1, 1]
                delta_z = np.clip(delta_z, -step_size, step_size)
                action[2] = delta_z / step_size
                action[2] = np.clip(action[2], -1.0, 1.0)
                action[3] = 0.0  # gripper打开
                
                # 到达block上方后，等待几步再进入抓取阶段
                if abs(ee_pos[2] - target_z) < 0.02:  # 在目标高度±2cm范围内
                    if not hasattr(self, '_approach_wait_steps'):
                        self._approach_wait_steps = 0
                    self._approach_wait_steps += 1
                    if self._approach_wait_steps >= 1:
                        self.phase = "grasp"
                        self.grasp_step = 0
                        if hasattr(self, '_approach_wait_steps'):
                            delattr(self, '_approach_wait_steps')
                        logger.info(f"  [Controller] ✅ 到达block上方 (ee_z={ee_pos[2]:.3f}, target_z={target_z:.3f})，进入抓取阶段")
                
        elif self.phase == "grasp":
            # 阶段2: 下降并抓取block（block位置固定：[0.5, 0.0, 0.06]）
            # 关键修复：必须真正下降到block位置（ee_z接近block_z，约1cm），确保能抓取到block
            # 成功条件要求：dist < 0.05（TCP和block的3D距离<5cm），所以必须下降到block上方1cm以内
            # 这样才能确保gripper关闭时能真正抓取到block，而不是"抓空气"
            target_z = cube_pos[2] + 0.01  # block上方1cm（确保能抓取到block）
            dist_z = ee_pos[2] - target_z  # ee在block上方，需要下降（正值表示需要下降）
            
            # 关键修复：强制下降检查 - 如果ee_z > block_z + 0.01，必须继续下降
            # 防止因为数值误差或计算错误导致提前关闭gripper
            must_descend = ee_pos[2] > cube_pos[2] + 0.01
            
            # 关键修复：最小下降步数检查 - 至少下降5步才能关闭gripper
            # 防止第一次进入grasp阶段时立即关闭gripper，确保真正下降到block位置
            # 增加步数要求，确保有足够的时间下降到block位置，避免在中间高度就关闭gripper
            min_descend_steps = 5
            has_descended_enough = self.grasp_step >= min_descend_steps
            
            # 关键修复：更严格的下降检查 - 必须真正下降到block位置（ee_z在block_z上方1cm以内）
            # 并且必须满足：1) 需要下降 或 2) 还没下降够步数
            # 这确保不会在中间高度就关闭gripper（"抓空气"）
            if dist_z > 0.01 or (must_descend and not has_descended_enough):
                # 继续下降
                delta_z = target_z - ee_pos[2]  # 负值，表示向下
                # 修复：移除multiplier限制，让delta_z能够达到完整的归一化范围[-1, 1]
                # 这样能匹配人工数据的delta_z分布（范围[-1, 1]）
                delta_z = np.clip(delta_z, -step_size, step_size)
                action[2] = delta_z / step_size
                action[2] = np.clip(action[2], -1.0, 1.0)
                action[3] = 0.0  # 下降过程中gripper保持打开
                self.grasp_step += 1
                
                # 调试输出：每步都输出，确保能看到下降过程
                logger.info(f"  [Controller] {self.phase} (step={self.grasp_step}): "
                          f"下降中 ee_z={ee_pos[2]:.3f}, block_z={cube_pos[2]:.3f}, target_z={target_z:.3f}, dist_z={dist_z:.3f}, action[2]={action[2]:.3f}, "
                          f"must_descend={must_descend}, has_descended_enough={has_descended_enough}")
            else:
                # 已经到达block位置（ee_z在block_z上方1cm以内），关闭gripper
                # 关键修复1：添加gripper中间状态(1)，实现0->1->2的平滑过渡（匹配人工数据）
                # 关键修复2：减少等待时间，提高delta_z活跃度（从10步减少到3步）
                if not hasattr(self, '_grasp_close_wait_start'):
                    # 第一次到达block位置，开始关闭gripper（先设置中间状态1.0）
                    action[2] = 0.0  # 停止下降
                    action[3] = 1.0  # 关键修复：先设置中间状态(1)，而不是直接跳到2.0
                    self._grasp_close_wait_start = self.grasp_step
                    self._grasp_close_wait_steps = 2  # 关键修复：进一步减少等待时间（从3步到2步），匹配人工数据episode长度13.3 frames
                    self._gripper_transition_step = 0  # 用于跟踪gripper状态转换
                    logger.info(f"  [Controller] ✅ 到达block位置 (ee_z={ee_pos[2]:.3f}, block_z={cube_pos[2]:.3f}, target_z={target_z:.3f}, dist_z={dist_z:.3f}, grasp_step={self.grasp_step})，开始关闭gripper（中间状态1.0）")
                else:
                    wait_steps = self.grasp_step - self._grasp_close_wait_start
                    self._gripper_transition_step = wait_steps
                    
                    if wait_steps < 1:
                        # 第1步：保持中间状态(1)
                        action[2] = 0.0  # 保持停止
                        action[3] = 1.0  # 中间状态
                        logger.info(f"  [Controller] Gripper中间状态(1) ({wait_steps}/{self._grasp_close_wait_steps})")
                    elif wait_steps < self._grasp_close_wait_steps:
                        # 第2步：过渡到关闭状态(2)
                        action[2] = 0.0  # 保持停止
                        action[3] = 2.0  # 关闭gripper
                        logger.info(f"  [Controller] Gripper关闭中(2) ({wait_steps}/{self._grasp_close_wait_steps})")
                    else:
                        # gripper已关闭，立即开始lift（不等待）
                        action[3] = 2.0  # 保持gripper关闭
                        self.phase = "lift"
                        self._lift_log_step = 0
                        # 清理grasp阶段的等待变量
                        if hasattr(self, '_grasp_close_wait_start'):
                            delattr(self, '_grasp_close_wait_start')
                        logger.info(f"  [Controller] ✅ Gripper已关闭，立即开始lift (ee_z={ee_pos[2]:.3f}, block_z={cube_pos[2]:.3f})")
                        # 立即开始向上移动（不等待）
                        target_lift_z = cube_pos[2] + 0.15  # 提升15cm
                        delta_z = target_lift_z - ee_pos[2]
                        # 修复：移除multiplier限制，让delta_z能够达到完整的归一化范围[-1, 1]
                        # 这样能匹配人工数据的delta_z分布（范围[-1, 1]）
                        delta_z = np.clip(delta_z, -step_size, step_size)
                        action[2] = delta_z / step_size
                        action[2] = np.clip(action[2], -1.0, 1.0)
                
                self.grasp_step += 1
                
        elif self.phase == "lift":
            # 阶段3: 向上举起block（保持gripper关闭）
            # 关键修复：确保gripper在大部分时间都是关闭的（匹配人工数据mean=1.92）
            self._lift_log_step = getattr(self, '_lift_log_step', 0) + 1
            
            # 记录初始block Z位置（用于lift目标计算）
            if self.initial_cube_z is None:
                self.initial_cube_z = cube_pos[2]
            
            # 关键修复：使用环境的success检测逻辑（与环境完全一致）
            # 环境的success检测：dist < 0.05 and lift > 0.1
            # 使用tcp_pos而不是ee_pos，使用_z_init而不是self.initial_cube_z
            try:
                # 获取TCP位置（与环境一致）
                tcp_pos = self.base_env._data.sensor("2f85/pinch_pos").data.copy()
                # 获取环境的_z_init（如果可用）
                env_z_init = self.env_z_init if self.env_z_init is not None else self.initial_cube_z
                
                # 使用环境的success检测逻辑
                dist_to_block = np.linalg.norm(cube_pos[:3] - tcp_pos[:3])
                block_lift = cube_pos[2] - env_z_init
                is_success = dist_to_block < 0.05 and block_lift > 0.1
            except Exception as e:
                # Fallback：如果无法获取tcp_pos或_z_init，使用简化逻辑
                logger.debug(f"  [Controller] 无法获取tcp_pos或_z_init，使用简化逻辑: {e}")
                dist_to_block = np.linalg.norm(cube_pos[:3] - ee_pos[:3])
                block_lift = cube_pos[2] - (self.env_z_init if self.env_z_init is not None else self.initial_cube_z)
                is_success = dist_to_block < 0.05 and block_lift > 0.1
                tcp_pos = ee_pos  # 使用ee_pos作为fallback
            
            # 关键修复：在接近success条件时，减少保持时间，提高delta_z活跃度
            # 人工采集时，在接近success时可能已经停止移动，等待按下success按钮
            # 但为了匹配人工数据的delta_z活跃度（停止时间20-30%），需要减少保持时间
            # 注意：一旦环境检测到success，episode会立即结束（terminate_on_success=True）
            
            # 如果接近success条件（但还没完全满足），短暂保持
            is_near_success = dist_to_block < 0.08 and block_lift > 0.08  # 稍微放宽条件
            
            if is_near_success or is_success:
                if not self.success_detected:
                    self.success_detected = True
                    self.success_hold_steps = 0
                    logger.info(f"  [Controller] ✅ 接近/达到success条件 (lift={block_lift:.3f}m, dist={dist_to_block:.3f}m)，开始保持阶段")
                
                self.success_hold_steps += 1
                # 关键修复：完全移除保持时间，立即终止（匹配人工数据episode长度13.3 frames）
                # 人工采集时，一旦达到success条件，episode立即终止（terminate_on_success=True）
                # 不需要额外的保持时间，这样可以匹配人工数据的短episode长度
                # 继续向上移动直到环境自动终止（提高delta_z活跃度）
                target_z = self.initial_cube_z + 0.15
                delta_z = target_z - ee_pos[2]
                # 即使接近success，也继续移动（匹配人工数据的活跃度，mean=0.20, std=0.81）
                delta_z = np.clip(delta_z, -step_size, step_size)
                action[2] = delta_z / step_size
                action[2] = np.clip(action[2], -1.0, 1.0)
                action[3] = 2.0  # 保持gripper关闭
            else:
                # 持续向上移动，直到达到目标高度（block初始位置+15cm）
                target_z = self.initial_cube_z + 0.15
                delta_z = target_z - ee_pos[2]
                # 修复：移除multiplier限制，让delta_z能够达到完整的归一化范围[-1, 1]
                # 这样能匹配人工数据的delta_z分布（范围[-1, 1]）
                delta_z = np.clip(delta_z, -step_size, step_size)
                action[2] = delta_z / step_size
                action[2] = np.clip(action[2], -1.0, 1.0)
                action[3] = 2.0  # 保持gripper关闭（关键：确保gripper在大部分时间都是关闭的）
        
        return action.astype(np.float32)


def verify_video_file(video_path: Path, max_attempts: int = 3) -> bool:
    """验证视频文件是否完整且可解码"""
    for attempt in range(max_attempts):
        if not video_path.exists():
            if attempt < max_attempts - 1:
                time.sleep(0.2)
                continue
            logger.warning(f"视频文件不存在: {video_path}")
            return False
        
        try:
            with av.open(str(video_path), "r") as container:
                if len(container.streams.video) == 0:
                    logger.warning(f"视频文件没有视频流: {video_path}")
                    return False
                
                video_stream = container.streams.video[0]
                frame_count = 0
                for frame in container.decode(video_stream):
                    frame_count += 1
                    if frame_count >= 1:
                        break
                
                if frame_count == 0:
                    logger.warning(f"无法从视频文件解码任何帧: {video_path}")
                    return False
                
                file_size = video_path.stat().st_size
                if file_size == 0:
                    logger.warning(f"视频文件大小为0: {video_path}")
                    return False
                
                logger.debug(f"✅ 视频文件验证成功: {video_path}")
                return True
                
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.warning(f"视频文件验证失败（尝试 {attempt + 1}/{max_attempts}）: {e}，等待后重试...")
                time.sleep(0.3)
                continue
            logger.error(f"视频文件验证失败: {video_path}, 错误: {type(e).__name__}: {e}")
            return False
    
    return False


def verify_episode_videos(dataset, episode_index: int) -> bool:
    """验证episode的所有视频文件是否完整"""
    if not hasattr(dataset, 'meta') or dataset.meta.episodes is None:
        return True
    
    if episode_index >= len(dataset.meta.episodes):
        logger.warning(f"Episode {episode_index} 的元数据不存在")
        return False
    
    episode_meta = dataset.meta.episodes[episode_index]
    all_valid = True
    
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
    enable_camera_display: bool = False,
    camera_display_freq: int = 5,
    use_ft: bool = False,  # 新增参数：是否使用FT环境
):
    """
    自动化采集数据集
    
    Args:
        config_path: 配置文件路径
        num_episodes: 要采集的episode数量
        output_dir: 输出目录（如果不指定，会自动生成）
        enable_camera_display: 是否启用相机显示
        camera_display_freq: 相机显示频率
        use_ft: 是否使用FT（Force/Torque）环境版本
    """
    from lerobot.rl.acfql.gym_manipulator import GymManipulatorConfig
    from lerobot.rl.gym_manipulator import DatasetConfig
    import json
    import draccus
    
    # 读取JSON配置
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 移除不支持的字段
    # 1. env.type字段（draccus choice class不需要这个字段，它通过@EnvConfig.register_subclass自动识别）
    if 'env' in config_dict and 'type' in config_dict['env']:
        env_type = config_dict['env'].pop('type')
        logger.debug(f"移除了env.type字段: {env_type}（draccus会自动识别）")
    
    # 2. dataset.use_imagenet_stats字段（gym_manipulator的DatasetConfig不支持此字段）
    if 'dataset' in config_dict and 'use_imagenet_stats' in config_dict['dataset']:
        use_imagenet_stats = config_dict['dataset'].pop('use_imagenet_stats')
        logger.debug(f"移除了dataset.use_imagenet_stats字段: {use_imagenet_stats}")
    
    # 3. 确保dataset配置中有task字段（DatasetConfig必需字段）
    if 'dataset' in config_dict:
        if 'task' not in config_dict['dataset']:
            # 如果配置文件中没有task，尝试从env.task获取，或根据use_ft设置默认值
            if 'env' in config_dict and 'task' in config_dict['env']:
                config_dict['dataset']['task'] = config_dict['env']['task']
                logger.debug(f"从env.task获取task字段: {config_dict['dataset']['task']}")
            else:
                # 根据use_ft参数设置默认task
                default_task = f'PandaPickCube{"Ft" if use_ft else ""}Gamepad-v0'
                config_dict['dataset']['task'] = default_task
                logger.debug(f"设置默认task字段: {default_task}")
    
    # 4. 只保留GymManipulatorConfig支持的字段（env, dataset, mode, device）
    # 过滤掉训练相关的字段（output_dir, job_name, resume, seed, num_workers, batch_size, steps, log_freq, save_checkpoint, save_freq, wandb, policy等）
    gym_manipulator_config_dict = {}
    if 'env' in config_dict:
        gym_manipulator_config_dict['env'] = config_dict['env']
    if 'dataset' in config_dict:
        gym_manipulator_config_dict['dataset'] = config_dict['dataset']
    if 'mode' in config_dict:
        gym_manipulator_config_dict['mode'] = config_dict['mode']
    if 'device' in config_dict:
        gym_manipulator_config_dict['device'] = config_dict['device']
    
    # 使用draccus解析配置文件
    import sys
    import tempfile
    import os
    original_argv = sys.argv
    tmp_config_path = None
    try:
        # 创建临时配置文件（只包含GymManipulatorConfig支持的字段）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(gym_manipulator_config_dict, tmp_file, indent=4)
            tmp_config_path = tmp_file.name
        
        sys.argv = ['auto_collect_pick_lift_data.py', f'--config_path={tmp_config_path}']
        cfg = draccus.parse(config_class=GymManipulatorConfig, config_path=tmp_config_path, args=[])
    finally:
        sys.argv = original_argv
        # 清理临时文件
        if tmp_config_path and os.path.exists(tmp_config_path):
            os.unlink(tmp_config_path)
    
    # 设置mode为record
    cfg.mode = "record"
    
    # 根据use_ft参数选择环境任务
    if use_ft:
        base_task = "PandaPickCubeFtBase-v0"
        task_suffix = "ft"
    else:
        base_task = "PandaPickCubeBase-v0"
        task_suffix = ""
    
    # 设置数据集参数
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(__file__).parent.parent / "datasets" / "pick_lift"
        output_dir = f"{base_dir}/franka_sim_pick_lift_acfql{'_ft' if use_ft else ''}_{timestamp}"
    
    # 确保dataset配置存在
    if not hasattr(cfg, 'dataset') or cfg.dataset is None:
        from omegaconf import OmegaConf
        cfg.dataset = OmegaConf.structured(DatasetConfig(
            repo_id=config_dict.get('dataset', {}).get('repo_id', f'franka_sim_pick_lift_acfql{task_suffix}'),
            root=output_dir,
            task=config_dict.get('dataset', {}).get('task', f'PandaPickCube{"Ft" if use_ft else ""}Gamepad-v0'),
            num_episodes_to_record=num_episodes,
            push_to_hub=False,
        ))
    else:
        cfg.dataset.root = output_dir
        cfg.dataset.num_episodes_to_record = num_episodes
        cfg.dataset.push_to_hub = False
    
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"将采集 {num_episodes} 个episodes")
    logger.info(f"使用环境: {base_task} ({'FT版本' if use_ft else '非FT版本'})")
    
    # 创建Base环境（没有InputsControlWrapper）
    import gym_hil  # noqa: F401
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True
    gripper_penalty = cfg.env.processor.gripper.gripper_penalty if cfg.env.processor.gripper is not None else 0.0
    
    # 创建Base环境
    # 注意：只有FT版本支持include_velocity参数
    env_kwargs = {
        "image_obs": True,
        "render_mode": "human",
    }
    # 只有FT版本支持include_velocity参数，非FT版本不支持
    # 确保非FT环境不传递include_velocity参数（即使有默认值也要移除）
    if use_ft:
        env_kwargs["include_velocity"] = True  # FT环境需要include_velocity
    # 非FT版本不传递include_velocity参数（确保不包含这个键）
    elif "include_velocity" in env_kwargs:
        del env_kwargs["include_velocity"]  # 确保非FT环境不包含这个参数
    
    base_env = gym.make(f"gym_hil/{base_task}", **env_kwargs)
    
    # 手动应用必要的wrapper
    if use_gripper:
        base_env = GripperPenaltyWrapper(base_env, penalty=gripper_penalty)
    
    ee_step_size = DEFAULT_EE_STEP_SIZE
    base_env = EEActionWrapper(
        base_env, 
        ee_action_step_size=ee_step_size, 
        use_gripper=True
    )
    
    base_env = PassiveViewerWrapper(base_env, show_left_ui=True, show_right_ui=True)
    
    reset_delay = cfg.env.processor.reset.reset_time_s if cfg.env.processor.reset is not None else 1.0
    base_env = ResetDelayWrapper(base_env, delay_seconds=reset_delay)
    
    terminate_on_success = cfg.env.processor.reset.terminate_on_success if cfg.env.processor.reset is not None else True
    unwrapped_env = base_env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    if hasattr(unwrapped_env, '_terminate_on_success'):
        unwrapped_env._terminate_on_success = terminate_on_success
        logger.info(f"设置 terminate_on_success = {terminate_on_success}")
    
    env = base_env
    env_processor, action_processor = make_processors(env, None, cfg.env, cfg.device)
    
    # 确保InterventionActionProcessorStep也使用正确的terminate_on_success设置
    # 因为InterventionActionProcessorStep有自己的terminate_on_success参数，需要手动设置
    for step in action_processor.steps:
        if hasattr(step, 'terminate_on_success'):
            step.terminate_on_success = terminate_on_success
            logger.info(f"设置 InterventionActionProcessorStep.terminate_on_success = {terminate_on_success}")
    
    # 创建自动控制器
    controller = AutomaticPickLiftController(env)
    
    # 获取action维度
    action_dim = env.action_space.shape[0]
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper else False
    
    # 使用与gym_manipulator相同的数据集创建方式
    obs, info = env.reset()
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)
    
    # 构建features字典
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
            val_shape = value.squeeze(0).shape if isinstance(value, torch.Tensor) else np.array(value).shape
            features[key] = {
                "dtype": "float32",
                "shape": val_shape,
                "names": None,
            }
    
    # 创建数据集
    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.env.fps,
        root=cfg.dataset.root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        batch_encoding_size=1,
        features=features,
    )
    
    # 控制循环
    dt = 1.0 / cfg.env.fps
    episode_idx = 0
    
    # 统计信息
    episode_lengths = []
    episode_successes = []
    
    display_camera_views = enable_camera_display and isinstance(obs, dict) and "pixels" in obs
    camera_display_counter = 0
    if display_camera_views:
        import cv2
        cv2.namedWindow("front", cv2.WINDOW_NORMAL)
        cv2.namedWindow("wrist", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("front", 256, 256)
        cv2.resizeWindow("wrist", 256, 256)
        logger.info(f"📹 Camera views initialized (显示频率: 每{camera_display_freq}帧)")
    else:
        logger.info("📹 Camera display disabled (recommended for stable data collection)")
    
    while episode_idx < num_episodes:
        # Reset环境
        obs, info = env.reset()
        env_processor.reset()
        action_processor.reset()
        controller.reset()
        camera_display_counter = 0
        
        transition = create_transition(observation=obs, info=info)
        transition = env_processor(transition)
        
        episode_start_time = time.perf_counter()
        episode_step = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始采集 Episode {episode_idx + 1}/{num_episodes}")
        logger.info(f"{'='*60}")
        
        # 定义常量：与DEFAULT_EE_STEP_SIZE和EEActionWrapper一致
        ee_step_size_value = 0.025  # 修复：在循环外定义，避免变量名冲突
        
        while True:
            step_start_time = time.perf_counter()
            
            # 关键修复：在执行action之前，保存当前的observation（用于推断gripper action）
            # 这与人工采集的逻辑一致：从执行前的observation.state[14]推断gripper状态
            prev_observations = None
            if use_gripper:
                prev_observations = {
                    k: v.squeeze(0).cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in transition[TransitionKey.OBSERVATION].items()
                }
            
            # 从控制器获取动作
            controller_action = controller.get_action()  # numpy array: [delta_x/step_size, delta_y/step_size, delta_z/step_size, gripper]
            
            # 调试：检查reset后第一个action的gripper值
            if episode_step == 0:
                logger.info(f"  [Debug] Reset后第一个action (控制器输出): {controller_action}, gripper={controller_action[3] if len(controller_action) >= 4 else 'N/A'}, phase={controller.phase}")
            
            # 关键修复：保存执行前的teleop_action（归一化的numpy array格式，与人工采集一致）
            # 人工采集时，InputsControlWrapper在info["teleop_action"]中设置的是归一化的numpy array
            # step_env_and_process_transition会将其转换为tensor（带batch维度）并放入complementary_data
            # gym_manipulator.py会从complementary_data获取tensor，转换为numpy array后记录
            # 所以我们需要在complementary_data中设置tensor格式（带batch维度），而不是字典格式
            teleop_action_before_step = controller_action.copy() if isinstance(controller_action, np.ndarray) else np.array(controller_action)
            # 转换为tensor并添加batch维度（与step_env_and_process_transition的行为一致）
            teleop_action_tensor = torch.from_numpy(teleop_action_before_step).float()
            if teleop_action_tensor.dim() == 1:
                teleop_action_tensor = teleop_action_tensor.unsqueeze(0)
            
            # 转换为tensor用于执行
            if isinstance(controller_action, np.ndarray):
                action = torch.from_numpy(controller_action).float()
            else:
                action = torch.tensor(controller_action, dtype=torch.float32)
            
            # 执行动作
            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            
            # 关键修复：确保teleop_action是控制器生成的动作（执行前的动作，归一化的）
            # 问题：InterventionActionProcessorStep会覆盖complementary_data["teleop_action"]为处理后的action
            # 解决方案：在step_env_and_process_transition之后，强制设置teleop_action为控制器生成的动作
            # 这样确保teleop_action是控制器生成的动作，而不是处理后的action（虽然在没有intervention时应该一样，但为了保险起见）
            if TransitionKey.COMPLEMENTARY_DATA in transition:
                # 强制设置teleop_action为控制器生成的动作（执行前的动作，归一化的）
                # 这与人工采集的行为一致：teleop_action是实际执行的动作（在InputsControlWrapper中设置）
                # 对于自动采集，teleop_action应该是控制器生成的动作（执行前的动作）
                transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] = teleop_action_tensor
            
            obs = transition[TransitionKey.OBSERVATION]
            terminated = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)
            
            # 记录数据
            observations = {
                k: v.squeeze(0).cpu() if isinstance(v, torch.Tensor) else v
                for k, v in transition[TransitionKey.OBSERVATION].items()
            }
            
            # 关键修复：使用teleop_action作为记录的动作（与人工采集一致）
            # 人工采集时，teleop_action在complementary_data中是tensor格式（带batch维度）
            # gym_manipulator.py会将其转换为numpy array（归一化的）后记录
            # 这样可以确保数据格式完全一致
            action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                "teleop_action", transition[TransitionKey.ACTION]
            )
            
            # 转换为numpy array（与人工采集的处理方式一致）
            # 人工采集时，teleop_action是tensor格式（带batch维度），需要squeeze(0)转换为numpy array
            if isinstance(action_to_record, torch.Tensor):
                action_to_record = action_to_record.squeeze(0).cpu().numpy()  # 移除batch维度，转换为numpy array
            elif isinstance(action_to_record, dict):
                # 如果是字典格式（不应该出现，但为了兼容性保留），转换为归一化的numpy array
                action_to_record = np.array([
                    action_to_record.get("delta_x", 0.0) / ee_step_size_value,  # 重新归一化
                    action_to_record.get("delta_y", 0.0) / ee_step_size_value,
                    action_to_record.get("delta_z", 0.0) / ee_step_size_value,
                    action_to_record.get("gripper", 0.0)  # 临时值，后面会从observation推断
                ], dtype=np.float32)
            else:
                action_to_record = np.array(action_to_record) if hasattr(action_to_record, '__len__') else np.array([action_to_record])
            
            # 关键修复：从observation推断gripper值（与人工采集一致）
            # 人工采集时，gripper值是从observation.state[14]推断的，而不是直接使用gamepad输入
            # 自动采集也应该使用相同的推断逻辑，确保数据格式完全一致
            # 0.0表示完全打开，255.0表示完全关闭
            # 将其映射到action的0.0（打开）和2.0（关闭）
            # 注意：与非FT版本保持一致，对所有帧（包括第一帧）都使用相同的推断逻辑
            if use_gripper and len(action_to_record) >= 4:
                # 使用执行前的observation（prev_observations）而不是执行后的（observations）
                # 这样记录的action的gripper值对应的是执行前的状态，符合因果关系
                if prev_observations is not None:
                    state_obs = prev_observations.get("observation.state", None)
                else:
                    # Fallback: 如果无法获取执行前的状态，使用执行后的状态（但这不是最优的）
                    state_obs = observations.get("observation.state", None)
                    
                if state_obs is not None and len(state_obs) > 14:
                    real_gripper_state = state_obs[14].item() if isinstance(state_obs, torch.Tensor) else state_obs[14]
                    # 关键修复：使用与人工采集完全一致的推断逻辑（gym_manipulator.py:771-776）
                    # 人工采集的推断逻辑：
                    #   if real_gripper_state <= 1: action_to_record[3] = 0.0
                    #   elif real_gripper_state >= 200: action_to_record[3] = 2.0
                    #   else: action_to_record[3] = 1.0
                    # 使用完全相同的阈值，确保数据格式一致
                    if real_gripper_state <= 1:  # 接近0，认为是打开
                        action_to_record[3] = 0.0
                    elif real_gripper_state >= 200:  # 接近255，认为是关闭
                        action_to_record[3] = 2.0
                    else:  # 其他值，认为是中性
                        action_to_record[3] = 1.0
            
            # 调试：检查reset后第一个action记录到数据集的值
            if episode_step == 0:
                logger.info(f"  [Debug] 记录到数据集的action: {action_to_record}, gripper={action_to_record[3] if len(action_to_record) >= 4 else 'N/A'}")
            
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
            
            # 显示相机视图
            if display_camera_views:
                camera_display_counter += 1
                if camera_display_counter >= camera_display_freq:
                    camera_display_counter = 0
                    import cv2
                    front_img = transition[TransitionKey.OBSERVATION].get("observation.images.front")
                    wrist_img = transition[TransitionKey.OBSERVATION].get("observation.images.wrist")
                    
                    if front_img is not None:
                        if isinstance(front_img, torch.Tensor):
                            front_img = front_img.squeeze(0).cpu().numpy()
                        else:
                            front_img = np.asarray(front_img)
                        if len(front_img.shape) == 3 and front_img.shape[0] == 3:
                            front_img = np.transpose(front_img, (1, 2, 0))
                        if front_img.max() <= 1.0:
                            front_img = (front_img * 255).astype(np.uint8)
                        front_img_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("front", front_img_bgr)
                    
                    if wrist_img is not None:
                        if isinstance(wrist_img, torch.Tensor):
                            wrist_img = wrist_img.squeeze(0).cpu().numpy()
                        else:
                            wrist_img = np.asarray(wrist_img)
                        if len(wrist_img.shape) == 3 and wrist_img.shape[0] == 3:
                            wrist_img = np.transpose(wrist_img, (1, 2, 0))
                        if wrist_img.max() <= 1.0:
                            wrist_img = (wrist_img * 255).astype(np.uint8)
                        wrist_img_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("wrist", wrist_img_bgr)
                    
                    cv2.waitKey(1)
            
            # 检查episode结束
            if terminated or truncated:
                episode_time = time.perf_counter() - episode_start_time
                success = transition[TransitionKey.INFO].get("succeed", False)
                info = transition.get(TransitionKey.INFO, {})
                
                # 详细记录episode结束原因
                end_reason = []
                if terminated:
                    end_reason.append("terminated")
                if truncated:
                    end_reason.append("truncated")
                if success:
                    end_reason.append("success")
                
                logger.info(
                    f"Episode {episode_idx + 1} 结束: {episode_step} 步, "
                    f"{episode_time:.1f}秒, 成功: {success}, "
                    f"奖励: {transition[TransitionKey.REWARD]:.4f}, "
                    f"结束原因: {', '.join(end_reason)}, "
                    f"控制器阶段: {controller.phase}"
                )
                
                # 记录统计信息
                episode_lengths.append(episode_step)
                episode_successes.append(success)
                
                # 如果episode太短，记录警告
                if episode_step < 20:
                    logger.warning(
                        f"⚠️  Episode {episode_idx + 1} 异常短（{episode_step}步）！"
                        f"  terminated={terminated}, truncated={truncated}, success={success}, "
                        f"控制器阶段: {controller.phase}"
                    )
                
                try:
                    logger.info(f"正在保存 Episode {episode_idx + 1}...")
                    dataset.save_episode()
                    time.sleep(1.0)
                    
                    if hasattr(dataset, 'meta') and dataset.meta.episodes is not None:
                        if len(dataset.meta.episodes) > episode_idx:
                            logger.info(f"✅ Episode {episode_idx + 1} 元数据已保存")
                        else:
                            logger.warning(f"⚠️  Episode {episode_idx + 1} 元数据可能未完全写入")
                            time.sleep(0.5)
                    
                    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'video_keys') and len(dataset.meta.video_keys) > 0:
                        logger.info(f"正在验证 Episode {episode_idx + 1} 的视频文件...")
                        if verify_episode_videos(dataset, episode_idx):
                            logger.info(f"✅ Episode {episode_idx + 1} 保存成功（视频文件已验证）")
                        else:
                            logger.error(f"❌ Episode {episode_idx + 1} 的视频文件验证失败！")
                            raise RuntimeError(f"Episode {episode_idx + 1} 的视频文件损坏，停止采集")
                    else:
                        logger.info(f"✅ Episode {episode_idx + 1} 保存成功（无视频文件需要验证）")
                        
                except Exception as e:
                    logger.error(f"❌ Episode {episode_idx + 1} 保存失败: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                episode_idx += 1
                break
            
            # 超时检查
            if episode_step >= 250:
                logger.warning(f"Episode {episode_idx + 1} 超时，强制结束")
                try:
                    logger.info(f"正在保存 Episode {episode_idx + 1}（超时）...")
                    dataset.save_episode()
                    time.sleep(1.0)
                    
                    if hasattr(dataset, 'meta') and dataset.meta.episodes is not None:
                        if len(dataset.meta.episodes) > episode_idx:
                            logger.info(f"✅ Episode {episode_idx + 1} 元数据已保存（超时）")
                    
                    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'video_keys') and len(dataset.meta.video_keys) > 0:
                        logger.info(f"正在验证 Episode {episode_idx + 1} 的视频文件（超时）...")
                        if verify_episode_videos(dataset, episode_idx):
                            logger.info(f"✅ Episode {episode_idx + 1} 保存成功（超时，视频文件已验证）")
                        else:
                            logger.error(f"❌ Episode {episode_idx + 1} 的视频文件验证失败（超时）！")
                            raise RuntimeError(f"Episode {episode_idx + 1} 的视频文件损坏（超时），停止采集")
                    else:
                        logger.info(f"✅ Episode {episode_idx + 1} 保存成功（超时，无视频文件需要验证）")
                        
                except Exception as e:
                    logger.error(f"❌ Episode {episode_idx + 1} 保存失败（超时）: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                episode_idx += 1
                break
            
            # 维持fps
            busy_wait(dt - (time.perf_counter() - step_start_time))
    
    # 关闭数据集
    logger.info("停止图像写入器...")
    try:
        dataset.stop_image_writer()
        time.sleep(0.5)
        logger.info("✅ 图像写入器已停止")
    except Exception as e:
        logger.error(f"❌ 停止图像写入器失败: {e}")
        raise
    
    if hasattr(dataset, 'batch_encoding_size') and dataset.batch_encoding_size > 1:
        if hasattr(dataset, 'episodes_since_last_encoding') and dataset.episodes_since_last_encoding > 0:
            logger.info(f"编码剩余的 {dataset.episodes_since_last_encoding} 个episode的视频...")
            try:
                start_ep = dataset.num_episodes - dataset.episodes_since_last_encoding
                end_ep = dataset.num_episodes
                dataset._batch_save_episode_video(start_ep, end_ep)
                time.sleep(0.5)
                logger.info("✅ 剩余视频编码完成")
            except Exception as e:
                logger.error(f"❌ 批处理编码失败: {e}")
                raise
    
    logger.info("完成数据集写入（finalize）...")
    try:
        dataset.finalize()
        time.sleep(0.5)
        logger.info("✅ 数据集finalize成功")
    except Exception as e:
        logger.error(f"❌ 数据集finalize失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    logger.info(f"数据集已保存到: {output_dir}")
    logger.info(f"共采集 {episode_idx} 个episodes")
    
    # 输出统计信息
    if episode_lengths:
        total_frames = sum(episode_lengths)
        avg_length = total_frames / len(episode_lengths)
        min_length = min(episode_lengths)
        max_length = max(episode_lengths)
        success_rate = sum(episode_successes) / len(episode_successes) * 100
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 数据集统计信息:")
        logger.info(f"  总episodes: {len(episode_lengths)}")
        logger.info(f"  总frames: {total_frames}")
        logger.info(f"  平均episode长度: {avg_length:.1f} frames")
        logger.info(f"  最短episode: {min_length} frames")
        logger.info(f"  最长episode: {max_length} frames")
        logger.info(f"  成功率: {success_rate:.1f}%")
        logger.info(f"{'='*60}")
        
        # 如果平均长度太短，发出警告
        if avg_length < 25:
            logger.warning(
                f"⚠️  警告：平均episode长度过短（{avg_length:.1f} frames）！"
                f"正常应该为30-100 frames/episode。"
                f"请检查环境配置和控制器逻辑。"
            )
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="自动化采集pick lift数据集")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/simulation/acfql/gym_hil_env_fql_ft.json",
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
        "--use_ft",
        action="store_true",
        help="使用FT（Force/Torque）环境版本（默认使用非FT版本）",
    )
    parser.add_argument(
        "--enable_camera_display",
        action="store_true",
        help="启用相机可视化窗口（默认禁用）",
    )
    parser.add_argument(
        "--camera_display_freq",
        type=int,
        default=5,
        help="如果启用相机显示，每N帧显示一次（默认5）",
    )
    args = parser.parse_args()
    
    output_dir = auto_collect_dataset(
        config_path=args.config,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        enable_camera_display=args.enable_camera_display,
        camera_display_freq=args.camera_display_freq,
        use_ft=args.use_ft,
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

