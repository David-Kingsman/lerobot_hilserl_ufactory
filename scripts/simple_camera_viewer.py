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
    # 加载MuJoCo模型 - 使用KUKA pick plate场景
    xml_path = Path(__file__).parent.parent / "gym-hil" / "gym_hil" / "assets" / "kuka_pick_plate_scene.xml"
    
    if not xml_path.exists():
        print(f"❌ 找不到XML文件: {xml_path}")
        return
    
    print(f"📂 加载模型: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # 找到所有可用的相机
    available_cameras = {}
    for i in range(model.ncam):
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if cam_name:
            available_cameras[cam_name] = i
    
    print(f"\n📹 找到 {len(available_cameras)} 个相机:")
    for name, cam_id in available_cameras.items():
        print(f"   - {name} (ID: {cam_id})")
    
    # 默认选择front相机，如果没有则选择第一个
    if "front" in available_cameras:
        current_camera_name = "front"
    elif "handcam_rgb" in available_cameras:
        current_camera_name = "handcam_rgb"
    else:
        current_camera_name = list(available_cameras.keys())[0]
    
    current_camera_id = available_cameras[current_camera_name]
    print(f"\n✅ 当前选择相机: {current_camera_name} (ID: {current_camera_id})")
    
    # 初始化相机参数（从XML读取）
    camera_pos = model.cam_pos[current_camera_id].copy()
    camera_quat = model.cam_quat[current_camera_id].copy()
    
    # 保存所有相机的初始参数
    initial_camera_params = {}
    for name, cam_id in available_cameras.items():
        initial_camera_params[name] = {
            'pos': model.cam_pos[cam_id].copy(),
            'quat': model.cam_quat[cam_id].copy()
        }
    
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
    print("🎮 相机视角调整工具 (KUKA Pick Plate)")
    print("="*70)
    print("\n控制说明:")
    print("  鼠标: 左键旋转，中键平移，右键/滚轮缩放")
    print("\n  键盘调整相机参数:")
    print("    位置调整 (WASD + ZX):")
    print("      W/S: X轴前后移动 (W=向前, S=向后)")
    print("      A/D: Y轴左右移动 (A=左, D=右)")
    print("      Z/X: Z轴上下移动 (Z=上升⬆, X=下降⬇)")
    print("    角度调整 (IJKL + UO):")
    print("      I/K: 俯仰角 (上下看)")
    print("      J/L: 偏航角 (左右转)")
    print("      U/O: 翻滚角")
    print("    相机切换:")
    print("      F: 切换到front相机")
    print("      H: 切换到handcam_rgb (wrist)相机")
    print("    鼠标拖动模式:")
    print("      M: 切换到FREELOOK模式（可用鼠标拖动调整视角）")
    print("      C: 将当前viewer视角应用到当前FIXED相机")
    print("      V: 切换回FIXED模式（查看当前相机视角）")
    print("    其他:")
    print("      1/2: 切换精细/粗略模式 (精细=1mm/0.5°, 粗略=1cm/2°)")
    print("      P: 打印当前相机参数")
    print("      R: 重置当前相机到初始位置")
    print("      ESC/Q: 退出并显示所有相机的最终参数")
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
                               'p', 'r', '1', '2', 'q', 'f', 'h', 'm', 'c', 'v']:
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
        pos = model.cam_pos[current_camera_id]
        quat = model.cam_quat[current_camera_id]
        
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = rot.as_euler('xyz', degrees=True)
        
        R = rot.as_matrix()
        z_axis = R[:, 2]
        pitch = np.arcsin(-z_axis[2]) * 180 / np.pi
        
        print("\n" + "-"*70)
        print(f"📹 当前相机参数 ({current_camera_name}):")
        print(f"   位置 (pos): {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
        print(f"   四元数 (quat): {quat_to_string(quat)}")
        print(f"   欧拉角 (度): roll={euler[0]:.1f}, pitch={euler[1]:.1f}, yaw={euler[2]:.1f}")
        print(f"   俯仰角: {pitch:.1f}度 (向下为正)")
        print("-"*70 + "\n")
    
    # 使用MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置为FIXED相机模式
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = current_camera_id
        
        print("\n✅ 查看器已启动！使用FIXED相机模式")
        print(f"   当前相机: {current_camera_name}")
        print("   现在可以使用键盘调整相机参数，实时看到效果\n")
        
        # 相机模式：FIXED（查看FIXED相机）或FREELOOK（自由拖动）
        camera_mode = "FIXED"  # "FIXED" 或 "FREELOOK"
        
        is_running = True
        last_print_time = 0
        import time as time_module
        
        def switch_camera(camera_name):
            nonlocal current_camera_name, current_camera_id, camera_pos, camera_quat, camera_mode
            if camera_name in available_cameras:
                current_camera_id = available_cameras[camera_name]
                current_camera_name = camera_name
                camera_pos = model.cam_pos[current_camera_id].copy()
                camera_quat = model.cam_quat[current_camera_id].copy()
                # 更新viewer的相机ID，确保显示对应相机的视角
                viewer.cam.fixedcamid = current_camera_id
                # 强制同步viewer，确保视角立即更新
                viewer.sync()
                print(f"📹 切换到相机: {current_camera_name} (ID: {current_camera_id})")
                print_camera_info()  # 显示当前相机的参数
                return True
            else:
                print(f"❌ 找不到相机: {camera_name}")
                print(f"   可用相机: {list(available_cameras.keys())}")
                return False
        
        while viewer.is_running() and is_running:
            step_start = data.time
            
            # 处理键盘输入（单次触发模式）
            pos_changed = False
            quat_changed = False
            camera_switched = False
            from scipy.spatial.transform import Rotation
            
            # 处理队列中的按键事件（只处理一个事件，避免快速移动）
            try:
                char = key_event_queue.get_nowait()  # 只取一个事件，避免快速移动
                
                if char == 'esc' or char == 'q':
                    is_running = False
                    break
                elif char == 'f':
                    if switch_camera("front"):
                        camera_switched = True
                elif char == 'h':
                    if switch_camera("handcam_rgb"):
                        camera_switched = True
                        if camera_mode == "FIXED":
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            viewer.cam.fixedcamid = current_camera_id
                elif char == 'm':
                    # 切换到FREELOOK模式，可以用鼠标拖动
                    camera_mode = "FREELOOK"
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    print("🖱️  切换到FREELOOK模式 - 现在可以用鼠标拖动调整视角")
                    print("   左键拖动: 旋转视角")
                    print("   右键拖动: 平移视角")
                    print("   滚轮: 缩放")
                    print("   按 C 键将当前视角应用到当前FIXED相机")
                    print("   按 V 键切换回FIXED模式查看相机视角")
                elif char == 'v':
                    # 切换回FIXED模式，查看当前选择的相机
                    camera_mode = "FIXED"
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = current_camera_id
                    print(f"📹 切换回FIXED模式 - 查看 {current_camera_name} 相机视角")
                elif char == 'c':
                    # 捕获当前viewer的视角并应用到当前FIXED相机
                    if camera_mode == "FREELOOK":
                        from scipy.spatial.transform import Rotation
                        # 获取当前viewer的相机参数（FREELOOK模式的相机状态）
                        # viewer.cam.lookat是相机看向的点
                        lookat = viewer.cam.lookat.copy()
                        # viewer.cam.distance是相机到lookat的距离
                        distance = viewer.cam.distance
                        # viewer.cam.azimuth和elevation是球坐标系的角度
                        azimuth = viewer.cam.azimuth
                        elevation = viewer.cam.elevation
                        
                        # 将球坐标转换为笛卡尔坐标（相机位置）
                        # 球坐标: (distance, azimuth, elevation)
                        # azimuth: 方位角（水平旋转）
                        # elevation: 仰角（垂直角度）
                        cos_elev = np.cos(np.deg2rad(elevation))
                        sin_elev = np.sin(np.deg2rad(elevation))
                        cos_azim = np.cos(np.deg2rad(azimuth))
                        sin_azim = np.sin(np.deg2rad(azimuth))
                        
                        # 相机位置（相对于lookat的偏移）
                        camera_offset = np.array([
                            distance * cos_elev * sin_azim,
                            distance * cos_elev * cos_azim,
                            distance * sin_elev
                        ])
                        camera_pos_new = lookat + camera_offset
                        
                        # 构建相机的朝向（从lookat指向相机的方向）
                        forward = -camera_offset / distance  # 相机朝向lookat
                        # 使用MuJoCo的默认up向量作为参考
                        default_up = np.array([0, 0, 1])
                        right = np.cross(forward, default_up)
                        if np.linalg.norm(right) < 1e-6:
                            # 如果forward和up平行，使用另一个参考
                            right = np.array([1, 0, 0])
                        right = right / np.linalg.norm(right)
                        up = np.cross(right, forward)
                        up = up / np.linalg.norm(up)
                        
                        # 构建旋转矩阵（相机坐标系：right, up, -forward）
                        rot_matrix = np.array([
                            right,
                            up,
                            -forward
                        ]).T
                        
                        # 转换为四元数
                        rot = Rotation.from_matrix(rot_matrix)
                        camera_quat_new = rot.as_quat()  # [x, y, z, w]
                        camera_quat_new = np.array([camera_quat_new[3], camera_quat_new[0], camera_quat_new[1], camera_quat_new[2]])  # [w, x, y, z]
                        
                        # 应用到当前FIXED相机
                        model.cam_pos[current_camera_id] = camera_pos_new
                        model.cam_quat[current_camera_id] = camera_quat_new
                        
                        mujoco.mj_forward(model, data)
                        
                        print(f"✅ 已将当前viewer视角应用到 {current_camera_name} 相机")
                        print_camera_info()
                        pos_changed = True
                        quat_changed = True
                    else:
                        print("⚠️  请在FREELOOK模式下使用 C 键捕获视角")
                elif char == 'p':
                    print_camera_info()
                elif char == 'r':
                    # 重置到初始位置
                    initial = initial_camera_params[current_camera_name]
                    model.cam_pos[current_camera_id] = initial['pos'].copy()
                    model.cam_quat[current_camera_id] = initial['quat'].copy()
                    camera_pos = initial['pos'].copy()
                    camera_quat = initial['quat'].copy()
                    mujoco.mj_forward(model, data)
                    print(f"🔄 已重置 {current_camera_name} 到初始位置")
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
                    model.cam_pos[current_camera_id][0] += current_pos_step
                    pos_changed = True
                elif char == 's':
                    model.cam_pos[current_camera_id][0] -= current_pos_step
                    pos_changed = True
                elif char == 'a':
                    model.cam_pos[current_camera_id][1] += current_pos_step
                    pos_changed = True
                elif char == 'd':
                    model.cam_pos[current_camera_id][1] -= current_pos_step
                    pos_changed = True
                elif char == 'z':
                    model.cam_pos[current_camera_id][2] += current_pos_step
                    pos_changed = True
                elif char == 'x':
                    model.cam_pos[current_camera_id][2] -= current_pos_step
                    pos_changed = True
                # 角度调整
                elif char in ['i', 'k', 'j', 'l', 'u', 'o']:
                    current_quat = model.cam_quat[current_camera_id]
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
                        model.cam_quat[current_camera_id] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
                        
            except queue.Empty:
                pass  # 队列为空，没有新按键
            
            if pos_changed or quat_changed or camera_switched:
                mujoco.mj_forward(model, data)
                if camera_switched:
                    camera_pos = model.cam_pos[current_camera_id].copy()
                    camera_quat = model.cam_quat[current_camera_id].copy()
            
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
    
    # 退出时打印所有相机的最终参数
    print("\n" + "="*70)
    print("📹 所有相机的最终参数 (复制到XML):")
    print("="*70)
    
    from scipy.spatial.transform import Rotation
    
    output_lines = []
    for camera_name in sorted(available_cameras.keys()):
        cam_id = available_cameras[camera_name]
        final_pos = model.cam_pos[cam_id]
        final_quat = model.cam_quat[cam_id]
        
        rot = Rotation.from_quat([final_quat[1], final_quat[2], final_quat[3], final_quat[0]])
        euler = rot.as_euler('xyz', degrees=True)
        
        print(f'\n【{camera_name} 相机】')
        if camera_name == "front":
            print(f'<camera name="front" pos="{final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}" '
                  f'quat="{quat_to_string(final_quat)}" fovy="50"/>')
        elif camera_name == "handcam_rgb":
            print(f'<camera name="handcam_rgb" pos="{final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}" '
                  f'fovy="42.5" quat="{quat_to_string(final_quat)}"/>')
        else:
            print(f'<camera name="{camera_name}" pos="{final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}" '
                  f'quat="{quat_to_string(final_quat)}" fovy="45"/>')
        
        print(f'  位置 (pos): "{final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}"')
        print(f'  四元数 (quat): "{quat_to_string(final_quat)}"')
        print(f'  欧拉角 (度): roll={euler[0]:.1f}, pitch={euler[1]:.1f}, yaw={euler[2]:.1f}')
        
        R = rot.as_matrix()
        z_axis = R[:, 2]
        pitch = np.arcsin(-z_axis[2]) * 180 / np.pi
        print(f'  俯仰角: {pitch:.1f}度')
        
        output_lines.append(f"\n【{camera_name} 相机】\n")
        if camera_name == "front":
            output_lines.append(f'<camera name="front" pos="{final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}" '
                              f'quat="{quat_to_string(final_quat)}" fovy="50"/>\n')
        elif camera_name == "handcam_rgb":
            output_lines.append(f'<camera name="handcam_rgb" pos="{final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}" '
                              f'fovy="42.5" quat="{quat_to_string(final_quat)}"/>\n')
        else:
            output_lines.append(f'<camera name="{camera_name}" pos="{final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}" '
                              f'quat="{quat_to_string(final_quat)}" fovy="45"/>\n')
        output_lines.append(f'位置: {final_pos[0]:.3f} {final_pos[1]:.3f} {final_pos[2]:.3f}\n')
        output_lines.append(f'四元数: {quat_to_string(final_quat)}\n')
        output_lines.append(f'欧拉角: roll={euler[0]:.1f}, pitch={euler[1]:.1f}, yaw={euler[2]:.1f}\n')
        output_lines.append(f'俯仰角: {pitch:.1f}度\n')
    
    print("="*70 + "\n")
    
    # 保存到文件
    output_file = Path(__file__).parent / "camera_quaternion_output.txt"
    with open(output_file, 'w') as f:
        f.write("KUKA Pick Plate 相机参数\n")
        f.write("="*70 + "\n")
        f.writelines(output_lines)
    
    print(f"💾 所有相机参数已保存到: {output_file}")

if __name__ == "__main__":
    main()

