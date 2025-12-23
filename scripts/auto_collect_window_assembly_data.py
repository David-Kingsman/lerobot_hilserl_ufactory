#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动化采集window assembly任务数据的脚本，使用基于位置的简单控制器自动执行pick和insertion任务，替代人工操作
完全复用gym_manipulator的数据采集流程，只是用自动控制器替换gamepad输入，自动控制器会根据环境状态生成动作，并执行pick和insertion任务

使用示例：
# 采集window assembly数据（默认配置）
python scripts/auto_collect_window_assembly_data.py \
    --config configs/simulation/acfql/gym_hil_env_fql_kuka_window_assembly_6dof.json \
    --num_episodes 50 

# 启用相机显示（可选，默认禁用）
python scripts/auto_collect_window_assembly_data.py \
    --config configs/simulation/acfql/gym_hil_env_fql_kuka_window_assembly_6dof.json \
    --num_episodes 50 \
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
from scipy.spatial.transform import Rotation as R

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


class AutomaticWindowAssemblyController:
    """自动化控制器优化版：基于五阶段FSM的精密控制
    
    阶段A: Pre-Grasp - 预抓取位姿，考虑窗户倾斜角度
    阶段B: Suction - 沿法线方向下降并吸附
    阶段C: Lift & Clear - 安全抬升，避免碰撞
    阶段D: Align - 垂直对齐，使用平滑姿态插值
    阶段E: Insertion - 精密插入，最后2cm准静态推送
    """
    
    def __init__(self, env):
        self.env = env
        self.base_env = env
        while hasattr(self.base_env, 'env'):
            self.base_env = self.base_env.env
        if not hasattr(self.base_env, '_data') and hasattr(self.base_env, 'unwrapped'):
            self.base_env = self.base_env.unwrapped
        
        # 阶段定义：pre_grasp -> suction -> lift_clear -> align -> insertion
        self.phase = "pre_grasp"
        self.grasp_step = 0
        self._z_rotate_height = 0.55  # 提升到此高度后再开始旋转（需高于缺口中心0.4m以防扫地）
        
        # 渐进式插入阈值（多阶段）
        self._insertion_fast_threshold = 0.10   # 10cm以上：快速接近
        self._insertion_medium_threshold = 0.05  # 5-10cm：中速对齐
        self._insertion_precision_threshold = 0.02  # 2-5cm：慢速精密插入
        self._insertion_ultra_precision = 0.005  # 最后5mm：超精密推送
        
        self._window_tilt_angle = np.radians(10)  # 窗户倾斜角度约10度
        
        # 插入阶段的状态跟踪（用于误差补偿和卡住检测）
        self._insertion_start_pos = None
        self._insertion_last_pos = None
        self._insertion_stuck_counter = 0
        self._insertion_error_history = []  # 记录Y/Z误差历史，用于检测卡住
        
        # 验证是否能访问底层环境
        if hasattr(self.base_env, '_data'):
            logger.info("  [Controller] ✅ 成功访问底层环境，可以获取MuJoCo数据")
        else:
            logger.warning("  [Controller] ⚠️  无法访问底层环境，可能无法正确获取状态")
    
    def reset(self):
        """重置控制器状态"""
        self.phase = "pre_grasp"
        self.grasp_step = 0
        self._insertion_start_pos = None
        self._insertion_last_pos = None
        self._insertion_stuck_counter = 0
        self._insertion_error_history = []
        logger.info(f"  [Controller] 🔄 控制器已重置，当前阶段: {self.phase}")
    
    def _compute_window_normal(self, window_quat):
        """计算窗户法线方向（考虑倾斜角度）"""
        win_mat = R.from_quat(window_quat[[1,2,3,0]]).as_matrix()
        return win_mat[:, 2]  # 窗户Z轴（法线方向）
    
    def _smooth_rotation_action(self, current_quat, target_normal, target_width, max_angular_velocity=0.1):
        """计算平滑旋转动作，使用最短路径避免360度翻转"""
        win_mat = R.from_quat(current_quat[[1,2,3,0]]).as_matrix()
        curr_normal = win_mat[:, 2]  # 当前法线
        curr_width = win_mat[:, 0]   # 当前宽度轴
        
        # 计算旋转轴和角度
        rot_action = np.zeros(3)
        
        # RY旋转（pitch）：对齐法线
        normal_cross = np.cross(curr_normal, target_normal)
        normal_dot = np.clip(np.dot(curr_normal, target_normal), -1.0, 1.0)
        if abs(normal_dot) < 0.99:  # 如果还没对齐
            if np.linalg.norm(normal_cross) > 1e-6:
                normal_axis = normal_cross / np.linalg.norm(normal_cross)
                # 投影到Y轴（RY旋转）
                ry_component = np.dot(normal_axis, np.array([0, 1, 0]))
        normal_angle = np.arccos(normal_dot)
                rot_action[1] = np.clip(ry_component * normal_angle / max_angular_velocity, -1.0, 1.0)
        
        # RX旋转（roll）：对齐宽度
        width_cross = np.cross(curr_width, target_width)
        width_dot = np.clip(np.dot(curr_width, target_width), -1.0, 1.0)
        if abs(width_dot) < 0.99:  # 如果还没对齐
        if np.linalg.norm(width_cross) > 1e-6:
            width_axis = width_cross / np.linalg.norm(width_cross)
                # 投影到X轴（RX旋转）
                rx_component = np.dot(width_axis, np.array([1, 0, 0]))
                width_angle = np.arccos(width_dot)
                rot_action[0] = np.clip(rx_component * width_angle / max_angular_velocity, -1.0, 1.0)
        
        return rot_action
    
    def get_action(self):
        try:
            window_pos = self.base_env._data.sensor("window_pos").data.copy()
            window_quat = self.base_env._data.sensor("window_quat").data.copy()
            ee_pos = self.base_env._data.sensor("2f85/pinch_pos").data.copy()
            target_pos = self.base_env._data.site("target_site").xpos.copy()
        except Exception:
            return np.zeros(7, dtype=np.float32)

        action = np.zeros(7, dtype=np.float32)
        vacuum_on = self.base_env.get_gripper_pose()[0] > 127

        # --- 五阶段FSM状态机 ---

        if self.phase == "pre_grasp":
            # 阶段A: 预抓取位姿 - 移动到窗户正上方，考虑倾斜角度
            # 计算窗户法线方向（考虑倾斜）
            window_normal = self._compute_window_normal(window_quat)
            # 目标位置：窗户中心上方10-15cm，沿法线方向
            target_pre = window_pos + window_normal * 0.12
            diff = target_pre - ee_pos
            
            if np.linalg.norm(diff) < 0.02:
                self.phase = "suction"
                logger.info("  [Controller] 📍 阶段A完成：到达预抓取位姿，开始下降吸附")
            else:
                # 平滑移动到目标位置
                action[:3] = np.clip(diff / 0.05, -1, 1)

        elif self.phase == "suction":
            # 阶段B: 吸附 - 沿法线方向下降直到接触
            window_normal = self._compute_window_normal(window_quat)
            # 沿法线方向下降
            contact_target = window_pos - window_normal * 0.01  # 稍微穿透以确保接触
            diff = contact_target - ee_pos
            
            action[:3] = np.clip(diff / 0.03, -1, 1)  # 较慢的下降速度
            action[6] = 2.0  # 开启真空驱动
            
            if vacuum_on:
                self.grasp_step += 1
                if self.grasp_step > 15:  # 确保吸盘完全排气稳固
                    self.phase = "lift_clear"
                    logger.info("  [Controller] ✅ 阶段B完成：已吸附，开始安全抬升")

        elif self.phase == "lift_clear":
            # 阶段C: 安全抬升 - 沿倾斜背板的垂直方向抬起，避免碰撞
            target_z = self._z_rotate_height
            action[2] = 1.0  # 全力上升
            action[3:6] = 0.0  # 锁定旋转，只做垂直运动
            action[6] = 2.0
            
            # 核心检查：如果机械臂上去了但窗户没动，说明吸附失败
            if ee_pos[2] > 0.15 and window_pos[2] < 0.05:
                logger.warning("  [Controller] ❌ 吸附失败(窗户掉落)，重试...")
                self.phase = "pre_grasp"
                self.grasp_step = 0
            elif ee_pos[2] >= target_z - 0.02:
                self.phase = "align"
                logger.info("  [Controller] ⬆️ 阶段C完成：已安全抬升，开始姿态对齐")

        elif self.phase == "align":
            # 阶段D: 垂直对齐 - 移动到墙前方并调整姿态
            # 预插入验证：确保姿态和位置都完美对齐后再进入插入阶段
            target_normal = np.array([-1, 0, 0])  # 面向墙
            target_width = np.array([0, 1, 0])    # 宽度对齐Y轴
            
            win_mat = R.from_quat(window_quat[[1,2,3,0]]).as_matrix()
            normal = win_mat[:, 2]
            width_axis = win_mat[:, 0]
            
            # 更严格的姿态对齐检查（提高到0.995）
            normal_ok = np.dot(normal, target_normal) > 0.995
            width_ok = abs(np.dot(width_axis, target_width)) > 0.995
            
            # 移动到缺口前方5cm处，并稍微抬高（为插入做准备）
            # 在align阶段就稍微抬高，让窗户底部能够顺利滑入槽口
            pre_insert_lift = 0.02  # 提前抬高2cm
            target_entry = target_pos + np.array([-0.05, 0, pre_insert_lift])
            pos_diff = target_entry - window_pos
            pos_ok = np.linalg.norm(pos_diff) < 0.008  # 更严格的位置检查（8mm）
            
            # 检查Y/Z对齐（侧向误差）
            # 注意：target_entry已经包含了抬升高度，所以pos_diff[2]已经考虑了抬升
            lateral_error = np.sqrt(pos_diff[1]**2 + pos_diff[2]**2)
            lateral_ok = lateral_error < 0.008  # Y/Z误差小于8mm（考虑抬升高度）

            if normal_ok and width_ok and pos_ok and lateral_ok:
                self.phase = "insertion"
                logger.info("  [Controller] 🎯 阶段D完成：姿态和位置已完美对齐")
                logger.info(f"    - 法线对齐: {np.dot(normal, target_normal):.4f}")
                logger.info(f"    - 宽度对齐: {abs(np.dot(width_axis, target_width)):.4f}")
                logger.info(f"    - 位置误差: {np.linalg.norm(pos_diff)*1000:.1f}mm")
                logger.info(f"    - 侧向误差: {lateral_error*1000:.1f}mm")
                logger.info("  [Controller] ✅ 预插入验证通过，开始精密插入")
            else:
                action[6] = 2.0
                # 如果姿态还没对齐，先做旋转
                if not (normal_ok and width_ok):
                    # 使用平滑旋转动作，避免抖动
                    rot_action = self._smooth_rotation_action(
                        window_quat, target_normal, target_width, max_angular_velocity=0.08  # 降低角速度，更平滑
                    )
                    action[3:6] = rot_action  # RX, RY, RZ
                    # 旋转时保持位置稳定（轻微位置保持）
                    action[:3] = (target_entry - window_pos) * 0.5  # 缓慢移动到目标位置
            else:
                    # 姿态已对齐，精确移动到缺口前
                    # 优先修正Y/Z误差，再推进X
                    if lateral_error > 0.005:
                        # Y/Z误差较大，先修正侧向位置
                        action[1] = np.clip(pos_diff[1] * 10.0, -1, 1)
                        action[2] = np.clip(pos_diff[2] * 10.0, -1, 1)
                        action[0] = np.clip(pos_diff[0] * 3.0, -0.5, 0.5)  # 降低X推进速度
                    else:
                        # Y/Z已对齐，正常移动到目标位置
                        action[:3] = np.clip(pos_diff * 5.0, -1, 1)
                    action[3:6] = np.zeros(3)  # 保持姿态不变

        elif self.phase == "insertion":
            # 阶段E: 渐进式精密插入 - 多阶段自适应控制
            # 关键改进：插入时需要先抬起来，让窗户底部滑入槽口
            diff = target_pos - window_pos
            dist_to_target = np.linalg.norm(diff)
            x_error = abs(diff[0])
            y_error = abs(diff[1])
            z_error = abs(diff[2])
            
            # 计算插入进度（0=刚开始，1=完成）
            if self._insertion_start_pos is None:
                self._insertion_start_pos = window_pos.copy()
                logger.info("  [Controller] 🚀 开始渐进式插入流程")
            
            insertion_progress = 1.0 - (x_error / max(np.linalg.norm(self._insertion_start_pos - target_pos), 0.01))
            insertion_progress = np.clip(insertion_progress, 0.0, 1.0)
            
            # 动态调整目标Z位置：插入时需要稍微抬高
            # 策略：开始时抬高2-3cm，随着插入进度逐渐降低到目标高度
            lift_height = 0.03 * (1.0 - insertion_progress)  # 从3cm逐渐降到0
            adjusted_target_z = target_pos[2] + lift_height
            adjusted_diff_z = adjusted_target_z - window_pos[2]
            
            # 记录位置变化（用于检测卡住）
            if self._insertion_last_pos is not None:
                pos_change = np.linalg.norm(window_pos - self._insertion_last_pos)
                if pos_change < 0.001:  # 位置几乎没变化
                    self._insertion_stuck_counter += 1
                else:
                    self._insertion_stuck_counter = 0
            self._insertion_last_pos = window_pos.copy()
            
            # 记录Y/Z误差历史（用于检测误差增大）
            lateral_error = np.sqrt(y_error**2 + z_error**2)
            self._insertion_error_history.append(lateral_error)
            if len(self._insertion_error_history) > 10:
                self._insertion_error_history.pop(0)
            
            # 检测是否卡住（位置不变且误差增大）
            stuck = self._insertion_stuck_counter > 5
            error_increasing = (len(self._insertion_error_history) >= 5 and 
                               self._insertion_error_history[-1] > self._insertion_error_history[0] * 1.5)
            
            action[6] = 2.0
            
            # 渐进式多阶段插入策略
            if x_error < self._insertion_ultra_precision:
                # 最后5mm：超精密推送，极低速度
                max_speed = 0.2
                x_gain = 0.005
                lateral_gain = 25.0
                logger.debug(f"  [插入] 超精密模式: X误差={x_error*1000:.1f}mm")
                
            elif x_error < self._insertion_precision_threshold:
                # 2-5cm：精密插入模式，慢速推送
                max_speed = 0.4
                x_gain = 0.01
                lateral_gain = 22.0
                logger.debug(f"  [插入] 精密模式: X误差={x_error*1000:.1f}mm")
                
            elif x_error < self._insertion_medium_threshold:
                # 5-10cm：中速对齐模式，先对齐Y/Z再推进
                max_speed = 0.7
                # 如果Y/Z误差较大，先对齐再推进
                if lateral_error > 0.01:
                    x_gain = 0.005  # 降低X推进速度
                    lateral_gain = 20.0
                    logger.debug(f"  [插入] 中速对齐模式: 先修正Y/Z误差={lateral_error*1000:.1f}mm")
                else:
                    x_gain = 0.015
                    lateral_gain = 18.0
                    logger.debug(f"  [插入] 中速推进模式")
                    
            else:
                # 10cm以上：快速接近模式
                max_speed = 1.0
                x_gain = 0.02
                lateral_gain = 15.0
                logger.debug(f"  [插入] 快速接近模式: X误差={x_error*1000:.1f}mm")
            
            # 如果检测到卡住或误差增大，进行微调
            if stuck or error_increasing:
                logger.warning(f"  [插入] ⚠️ 检测到卡住或误差增大，进行微调...")
                # 稍微后退并重新对齐，同时稍微抬高
                action[0] = -0.3  # 轻微后退
                action[1] = np.clip(diff[1] * 25.0, -1, 1)  # 加强Y轴修正
                action[2] = np.clip(adjusted_diff_z * 20.0, -1, 1)  # 使用调整后的Z目标
                action[3:6] = np.zeros(3)  # 保持姿态
            else:
                # 正常插入动作
                # X轴推进（主方向）
                action[0] = np.clip(diff[0] / x_gain, -max_speed, max_speed)
                
                # Y轴修正（侧向误差，极其严格）
                action[1] = np.clip(diff[1] * lateral_gain, -1, 1)
                
                # Z轴修正：使用调整后的目标高度（插入时先抬起来）
                # 在插入初期，主动抬高；随着插入进度，逐渐降低到目标高度
                if insertion_progress < 0.3:
                    # 前30%插入：主动抬高，让窗户底部滑入
                    z_gain = lateral_gain * 1.2  # 稍微加强抬升力度
                    action[2] = np.clip(adjusted_diff_z * z_gain, -1, 1)
                    logger.debug(f"  [插入] 抬升阶段: 进度={insertion_progress*100:.1f}%, 抬升高度={lift_height*1000:.1f}mm")
                elif insertion_progress < 0.7:
                    # 30-70%插入：逐渐降低高度
                    z_gain = lateral_gain
                    action[2] = np.clip(adjusted_diff_z * z_gain, -1, 1)
                    logger.debug(f"  [插入] 过渡阶段: 进度={insertion_progress*100:.1f}%, 抬升高度={lift_height*1000:.1f}mm")
                else:
                    # 最后30%插入：精确对齐到目标高度
                    z_gain = lateral_gain
                    action[2] = np.clip(diff[2] * z_gain, -1, 1)  # 使用原始目标高度
                    logger.debug(f"  [插入] 精确阶段: 进度={insertion_progress*100:.1f}%")
                
                # 保持姿态不变（不旋转）
                action[3:6] = np.zeros(3)
            
            # 完成检查：X误差小于2mm且Y/Z误差小于3mm
            if x_error < 0.002 and lateral_error < 0.003:
                logger.info("  [Controller] 🏆 阶段E完成：安装成功！精度达标！")
                action[6] = 0.0  # 安装完成，释放吸盘
                # 保持位置，等待环境确认
                action[:3] = np.zeros(3)

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
):
    """
    自动化采集数据集
    
    Args:
        config_path: 配置文件路径
        num_episodes: 要采集的episode数量
        output_dir: 输出目录（如果不指定，会自动生成）
        enable_camera_display: 是否启用相机显示
        camera_display_freq: 相机显示频率
    """
    from lerobot.rl.acfql.gym_manipulator import GymManipulatorConfig
    from lerobot.rl.gym_manipulator import DatasetConfig
    import json
    import draccus
    
    # 读取JSON配置
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 移除不支持的字段
    if 'env' in config_dict and 'type' in config_dict['env']:
        env_type = config_dict['env'].pop('type')
        logger.debug(f"移除了env.type字段: {env_type}")
    
    if 'dataset' in config_dict and 'use_imagenet_stats' in config_dict['dataset']:
        use_imagenet_stats = config_dict['dataset'].pop('use_imagenet_stats')
        logger.debug(f"移除了dataset.use_imagenet_stats字段: {use_imagenet_stats}")
    
    # 确保dataset配置中有task字段
    if 'dataset' in config_dict:
        if 'task' not in config_dict['dataset']:
            if 'env' in config_dict and 'task' in config_dict['env']:
                config_dict['dataset']['task'] = config_dict['env']['task']
                logger.debug(f"从env.task获取task字段: {config_dict['dataset']['task']}")
            else:
                default_task = 'KukaPickWindowGamepad6DoF-v0'
                config_dict['dataset']['task'] = default_task
                logger.debug(f"设置默认task字段: {default_task}")
    
    # 只保留GymManipulatorConfig支持的字段
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(gym_manipulator_config_dict, tmp_file, indent=4)
            tmp_config_path = tmp_file.name
        
        sys.argv = ['auto_collect_window_assembly_data.py', f'--config_path={tmp_config_path}']
        cfg = draccus.parse(config_class=GymManipulatorConfig, config_path=tmp_config_path, args=[])
    finally:
        sys.argv = original_argv
        if tmp_config_path and os.path.exists(tmp_config_path):
            os.unlink(tmp_config_path)
    
    # 设置mode为record
    cfg.mode = "record"
    
    # 设置数据集参数
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(__file__).parent.parent / "datasets"
        output_dir = f"{base_dir}/kuka_sim_window_assembly_acfql_6dof_{timestamp}"
    
    # 确保dataset配置存在
    if not hasattr(cfg, 'dataset') or cfg.dataset is None:
        from omegaconf import OmegaConf
        cfg.dataset = OmegaConf.structured(DatasetConfig(
            repo_id=config_dict.get('dataset', {}).get('repo_id', 'kuka_sim_window_assembly_acfql_6dof'),
            root=output_dir,
            task=config_dict.get('dataset', {}).get('task', 'KukaPickWindowGamepad6DoF-v0'),
            num_episodes_to_record=num_episodes,
            push_to_hub=False,
        ))
    else:
        cfg.dataset.root = output_dir
        cfg.dataset.num_episodes_to_record = num_episodes
        cfg.dataset.push_to_hub = False
    
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"将采集 {num_episodes} 个episodes")
    
    # 创建Base环境（没有InputsControlWrapper）
    import gym_hil  # noqa: F401
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True
    gripper_penalty = cfg.env.processor.gripper.gripper_penalty if cfg.env.processor.gripper is not None else 0.0
    
    # 创建Base环境
    base_task = "KukaPickWindowBase-v0"
    base_env = gym.make(
        f"gym_hil/{base_task}",
        image_obs=True,
        render_mode="human",
    )
    
    # 手动应用必要的wrapper
    if use_gripper:
        base_env = GripperPenaltyWrapper(base_env, penalty=gripper_penalty)
    
    ee_step_size = DEFAULT_EE_STEP_SIZE
    base_env = EEActionWrapper(
        base_env, 
        ee_action_step_size=ee_step_size, 
        use_gripper=True,
        use_6dof=True  # 使用6-DoF控制
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
    for step in action_processor.steps:
        if hasattr(step, 'terminate_on_success'):
            step.terminate_on_success = terminate_on_success
            logger.info(f"设置 InterventionActionProcessorStep.terminate_on_success = {terminate_on_success}")
    
    # 创建自动控制器
    controller = AutomaticWindowAssemblyController(env)
    
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
        
        while True:
            step_start_time = time.perf_counter()
            
            # 从控制器获取动作
            controller_action = controller.get_action()
            
            # 保存执行前的teleop_action
            teleop_action_before_step = controller_action.copy() if isinstance(controller_action, np.ndarray) else np.array(controller_action)
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
            
            # 确保teleop_action是控制器生成的动作
            if TransitionKey.COMPLEMENTARY_DATA in transition:
                transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] = teleop_action_tensor
            
            obs = transition[TransitionKey.OBSERVATION]
            terminated = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)
            
            # 记录数据
            observations = {
                k: v.squeeze(0).cpu() if isinstance(v, torch.Tensor) else v
                for k, v in transition[TransitionKey.OBSERVATION].items()
            }
            
            # 使用teleop_action作为记录的动作
            action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                "teleop_action", transition[TransitionKey.ACTION]
            )
            
            # 转换为numpy array
            if isinstance(action_to_record, torch.Tensor):
                action_to_record = action_to_record.squeeze(0).cpu().numpy()
            else:
                action_to_record = np.array(action_to_record) if hasattr(action_to_record, '__len__') else np.array([action_to_record])
            
            # 从observation推断gripper值（与人工采集一致）
            if use_gripper and len(action_to_record) >= 7:
                state_obs = observations.get("observation.state", None)
                if state_obs is not None and len(state_obs) > 14:
                    real_gripper_state = state_obs[14].item() if isinstance(state_obs, torch.Tensor) else state_obs[14]
                    if real_gripper_state <= 1:
                        action_to_record[6] = 0.0
                    elif real_gripper_state >= 200:
                        action_to_record[6] = 2.0
                    else:
                        action_to_record[6] = 1.0
            
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
        
        if avg_length < 25:
            logger.warning(
                f"⚠️  警告：平均episode长度过短（{avg_length:.1f} frames）！"
                f"正常应该为30-100 frames/episode。"
                f"请检查环境配置和控制器逻辑。"
            )
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="自动化采集window assembly数据集")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/simulation/acfql/gym_hil_env_fql_kuka_window_assembly_6dof.json",
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

