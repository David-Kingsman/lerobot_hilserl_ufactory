#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版相机视角选择工具
直接在FIXED模式下调整相机参数，实时看到效果
"""

import sys
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import threading

# 添加gym_hil路径
sys.path.insert(0, str(Path(__file__).parent.parent / "gym-hil"))

def main():
    # 加载MuJoCo模型
    xml_path = Path(__file__).parent.parent / "gym-hil" / "gym_hil" / "assets" / "masonry_insertion.xml"
    
    if not xml_path.exists():
        print(f"❌ 找不到XML文件: {xml_path}")
        return
    
    print(f"📂 加载模型: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # 找到front相机的ID
    front_camera_id = None
    for i in range(model.ncam):
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if cam_name == "front":
            front_camera_id = i
            break
    
    if front_camera_id is None:
        print("❌ 找不到'front'相机")
        return
    
    print(f"✅ 找到front相机 (ID: {front_camera_id})")
    
    # 初始化相机参数（从XML读取）
    camera_pos = model.cam_pos[front_camera_id].copy()
    camera_quat = model.cam_quat[front_camera_id].copy()
    
    # 调整步长（可调）
    POS_STEP_FINE = 0.001  # 精细模式：1mm
    POS_STEP_COARSE = 0.01  # 粗略模式：1cm
    ANGLE_STEP_FINE = np.deg2rad(0.5)  # 精细模式：0.5度
    ANGLE_STEP_COARSE = np.deg2rad(2)  # 粗略模式：2度
    
    # 当前步长（可以用数字键切换）
    current_pos_step = POS_STEP_COARSE
    current_angle_step = ANGLE_STEP_COARSE
    fine_mode = False
    
    print("\n" + "="*70)
    print("🎮 简化版相机视角调整工具")
    print("="*70)
    print("\n控制说明:")
    print("  鼠标: 左键旋转，中键平移，右键/滚轮缩放")
    print("\n  键盘调整相机参数:")
    print("    位置调整 (WASD + ZX):")
    print("      W/S: X轴前后移动")
    print("      A/D: Y轴左右移动")
    print("      Z/X: Z轴上下移动 (Z=上, X=下)")
    print("    角度调整 (IJKL + UO, 步长1度):")
    print("      I/K: 俯仰角 (上下看)")
    print("      J/L: 偏航角 (左右转)")
    print("      U/O: 翻滚角")
    print("    其他:")
    print("      1/2: 切换精细/粗略模式 (精细=1mm/0.5°, 粗略=1cm/2°)")
    print("      P: 打印当前相机参数")
    print("      R: 重置到初始位置")
    print("      ESC/Q: 退出并显示最终参数")
    print("="*70 + "\n")
    
    # 使用队列存储按键事件（单次触发模式）
    import queue
    key_event_queue = queue.Queue()
    
    # 当前步长
    current_pos_step = POS_STEP_COARSE
    current_angle_step = ANGLE_STEP_COARSE
    
    # 启动键盘监听器
    try:
        from pynput import keyboard
        
        # 使用队列来存储按键事件，避免持续触发
        import queue
        key_event_queue = queue.Queue()
        
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char:
                    char = key.char.lower()
                    # 只处理单次触发的按键
                    if char in ['w', 's', 'a', 'd', 'z', 'x', 'i', 'k', 'j', 'l', 'u', 'o', 
                               'p', 'r', '1', '2', 'q']:
                        key_event_queue.put(char)
            except AttributeError:
                if key == keyboard.Key.esc:
                    key_event_queue.put('esc')
        
        def on_release(key):
            pass  # 不需要处理释放事件，使用单次触发
        
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        print("✅ 键盘监听器已启动")
    except ImportError:
        print("⚠️  警告: pynput未安装，键盘快捷键将不可用")
        print("   请安装: pip install pynput")
        listener = None
    
    def quat_to_string(quat):
        return f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
    
    def print_camera_info():
        from scipy.spatial.transform import Rotation
        pos = model.cam_pos[front_camera_id]
        quat = model.cam_quat[front_camera_id]
        
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = rot.as_euler('xyz', degrees=True)
        
        R = rot.as_matrix()
        z_axis = R[:, 2]
        pitch = np.arcsin(-z_axis[2]) * 180 / np.pi
        
        print("\n" + "-"*70)
        print("📹 当前相机参数:")
        print(f"   位置 (pos): {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
        print(f"   四元数 (quat): {quat_to_string(quat)}")
        print(f"   欧拉角 (度): roll={euler[0]:.1f}, pitch={euler[1]:.1f}, yaw={euler[2]:.1f}")
        print(f"   俯仰角: {pitch:.1f}度 (向下为正)")
        print("-"*70 + "\n")
    
    # 使用MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置为FIXED相机模式
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = front_camera_id
        
        print("\n✅ 查看器已启动！使用FIXED相机模式")
        print("   现在可以使用键盘调整相机参数，实时看到效果\n")
        
        is_running = True
        last_print_time = 0
        import time as time_module
        
        while viewer.is_running() and is_running:
            step_start = data.time
            
            # 处理键盘输入（单次触发模式）
            pos_changed = False
            quat_changed = False
            from scipy.spatial.transform import Rotation
            
            # 处理队列中的按键事件（只处理一个事件，避免快速移动）
            try:
                char = key_event_queue.get_nowait()  # 只取一个事件，避免快速移动
                
                if char == 'esc' or char == 'q':
                    is_running = False
                    break
                elif char == 'p':
                    print_camera_info()
                elif char == 'r':
                    model.cam_pos[front_camera_id] = camera_pos.copy()
                    model.cam_quat[front_camera_id] = camera_quat.copy()
                    mujoco.mj_forward(model, data)
                    print("🔄 已重置到初始位置")
                    pos_changed = True
                    quat_changed = True
                elif char == '1':
                    current_pos_step = POS_STEP_FINE
                    current_angle_step = ANGLE_STEP_FINE
                    print(f"🔧 切换到精细模式: 位置步长={current_pos_step*1000:.1f}mm, 角度步长={np.rad2deg(current_angle_step):.1f}°")
                elif char == '2':
                    current_pos_step = POS_STEP_COARSE
                    current_angle_step = ANGLE_STEP_COARSE
                    print(f"🔧 切换到粗略模式: 位置步长={current_pos_step*1000:.1f}mm, 角度步长={np.rad2deg(current_angle_step):.1f}°")
                # 位置调整
                elif char == 'w':
                    model.cam_pos[front_camera_id][0] += current_pos_step
                    pos_changed = True
                elif char == 's':
                    model.cam_pos[front_camera_id][0] -= current_pos_step
                    pos_changed = True
                elif char == 'a':
                    model.cam_pos[front_camera_id][1] += current_pos_step
                    pos_changed = True
                elif char == 'd':
                    model.cam_pos[front_camera_id][1] -= current_pos_step
                    pos_changed = True
                elif char == 'z':
                    model.cam_pos[front_camera_id][2] += current_pos_step
                    pos_changed = True
                elif char == 'x':
                    model.cam_pos[front_camera_id][2] -= current_pos_step
                    pos_changed = True
                # 角度调整
                elif char in ['i', 'k', 'j', 'l', 'u', 'o']:
                    current_quat = model.cam_quat[front_camera_id]
                    rot = Rotation.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
                    euler = rot.as_euler('xyz')
                    
                    if char == 'i':
                        euler[1] += current_angle_step  # pitch up
                        quat_changed = True
                    elif char == 'k':
                        euler[1] -= current_angle_step  # pitch down
                        quat_changed = True
                    elif char == 'j':
                        euler[2] += current_angle_step  # yaw left
                        quat_changed = True
                    elif char == 'l':
                        euler[2] -= current_angle_step  # yaw right
                        quat_changed = True
                    elif char == 'u':
                        euler[0] += current_angle_step  # roll
                        quat_changed = True
                    elif char == 'o':
                        euler[0] -= current_angle_step  # roll
                        quat_changed = True
                    
                    if quat_changed:
                        new_rot = Rotation.from_euler('xyz', euler)
                        new_quat = new_rot.as_quat()
                        model.cam_quat[front_camera_id] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
                        
            except queue.Empty:
                pass  # 队列为空，没有新按键
            
            if pos_changed or quat_changed:
                mujoco.mj_forward(model, data)
            
            # 物理仿真步进
            mujoco.mj_step(model, data)
            
            # 同步查看器
            viewer.sync()
            
            # 控制步进速度
            time_until_next_step = model.opt.timestep - (data.time - step_start)
            if time_until_next_step > 0:
                time_module.sleep(time_until_next_step)
    
    # 停止键盘监听器
    if listener is not None:
        listener.stop()
    
    # 退出时打印最终相机参数
    print("\n" + "="*70)
    print("📹 最终相机参数 (复制到XML):")
    print("="*70)
    final_pos = model.cam_pos[front_camera_id]
    final_quat = model.cam_quat[front_camera_id]
    
    print(f'\n<camera name="front" pos="{final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}" '
          f'quat="{quat_to_string(final_quat)}" fovy="45"/>')
    
    print("\n详细参数:")
    print(f'  位置 (pos): "{final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}"')
    print(f'  四元数 (quat): "{quat_to_string(final_quat)}"')
    
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat([final_quat[1], final_quat[2], final_quat[3], final_quat[0]])
    euler = rot.as_euler('xyz', degrees=True)
    print(f'  欧拉角 (度): roll={euler[0]:.1f}, pitch={euler[1]:.1f}, yaw={euler[2]:.1f}')
    
    R = rot.as_matrix()
    z_axis = R[:, 2]
    pitch = np.arcsin(-z_axis[2]) * 180 / np.pi
    print(f'  俯仰角: {pitch:.1f}度')
    print("="*70 + "\n")
    
    # 保存到文件
    output_file = Path(__file__).parent / "camera_quaternion_output.txt"
    with open(output_file, 'w') as f:
        f.write("相机参数\n")
        f.write("="*70 + "\n")
        f.write(f'<camera name="front" pos="{final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}" ')
        f.write(f'quat="{quat_to_string(final_quat)}" fovy="45"/>\n\n')
        f.write(f'位置: {final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}\n')
        f.write(f'四元数: {quat_to_string(final_quat)}\n')
        f.write(f'欧拉角: roll={euler[0]:.1f}, pitch={euler[1]:.1f}, yaw={euler[2]:.1f}\n')
        f.write(f'俯仰角: {pitch:.1f}度\n')
    
    print(f"💾 参数已保存到: {output_file}")

if __name__ == "__main__":
    main()

