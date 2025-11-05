#!/usr/bin/env python3
"""
ROI坐标换算工具
将在128x128图像上选择的ROI坐标换算到480x640原始图像尺寸
"""

def convert_roi_128_to_original(roi_128, original_height=480, original_width=640, resized_size=128):
    """
    将128x128图像上的ROI坐标换算到原始图像尺寸
    
    Args:
        roi_128: (top, left, height, width) 在128x128图像上的ROI
        original_height: 原始图像高度，默认480
        original_width: 原始图像宽度，默认640
        resized_size: resize后的尺寸，默认128
    
    Returns:
        (top, left, height, width) 在原始图像上的ROI坐标
    """
    top, left, height, width = roi_128
    
    # 计算缩放比例
    scale_x = original_width / resized_size   # 640/128 = 5.0
    scale_y = original_height / resized_size  # 480/128 = 3.75
    
    # 换算坐标
    original_top = int(top * scale_y)
    original_left = int(left * scale_x)
    original_height = int(height * scale_y)
    original_width = int(width * scale_x)
    
    return (original_top, original_left, original_height, original_width)

def print_conversion_info():
    """打印换算信息"""
    print("=" * 60)
    print("ROI坐标换算工具")
    print("=" * 60)
    print("原始图像尺寸: 640x480 (宽x高)")
    print("Resize后尺寸: 128x128")
    print("缩放比例:")
    print(f"  X方向 (宽度): 640/128 = {640/128}")
    print(f"  Y方向 (高度): 480/128 = {480/128}")
    print("-" * 60)

def main():
    print_conversion_info()
    
    # 示例：你之前获得的ROI坐标
    webcam_roi_128 = (41, 56, 31, 51)
    realsense_roi_128 = (50, 46, 28, 27)
    
    print("示例换算 (你之前的ROI坐标):")
    print(f"webcam_1 (128x128):   {webcam_roi_128}")
    webcam_original = convert_roi_128_to_original(webcam_roi_128)
    print(f"webcam_1 (640x480):   {webcam_original}")
    print()
    
    print(f"realsense (128x128):  {realsense_roi_128}")
    realsense_original = convert_roi_128_to_original(realsense_roi_128)
    print(f"realsense (640x480):  {realsense_original}")
    print()
    
    print("配置文件格式:")
    print('"crop_params_dict": {')
    print(f'    "observation.images.webcam_1": {list(webcam_original)},')
    print(f'    "observation.images.realsense": {list(realsense_original)}')
    print('}')
    print()
    
    # 交互式输入
    print("=" * 60)
    print("交互式换算 - 输入你的ROI坐标:")
    print("格式: top,left,height,width (用逗号分隔)")
    print("输入 'q' 退出")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n请输入ROI坐标 (128x128图像上): ").strip()
            if user_input.lower() == 'q':
                break
                
            # 解析输入
            coords = [int(x.strip()) for x in user_input.split(',')]
            if len(coords) != 4:
                print("错误: 请输入4个数值 (top,left,height,width)")
                continue
                
            roi_128 = tuple(coords)
            roi_original = convert_roi_128_to_original(roi_128)
            
            print(f"输入 (128x128):  {roi_128}")
            print(f"换算 (640x480):  {roi_original}")
            print(f"配置格式: {list(roi_original)}")
            
        except ValueError:
            print("错误: 请输入有效的数字")
        except KeyboardInterrupt:
            break
    
    print("\n换算完成!")

if __name__ == "__main__":
    main()
