#!/usr/bin/env python3
"""
LeRobot Crop Visualizer 

visualize the crop effect of LeRobot, including the original image and the cropped image comparison.
support reading crop parameters from the configuration file, and real-time adjustment of the crop region.

usage:
    python test/crop_visualizer.py --config_path configs/ufactory/env_config_hilserl_lite6.json

controls:
    - ESC: exit
    - SPACE: pause/continue
    - R: reset crop parameters to the default values in the configuration file
    - mouse drag: adjust the crop region (hold the left mouse button to drag)
    - scroll wheel: adjust the crop region size
"""

import argparse
import cv2
import json
import numpy as np
import threading
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import torchvision.transforms.functional as F

class CropVisualizer:
    def __init__(self, config_path: str):
        """initialize the crop visualizer"""
        self.config_path = config_path
        self.crop_params = self.load_crop_params()
        self.original_images = {}
        self.cropped_images = {}
        self.running = True
        self.paused = False
        self.camera_threads = {}
        self.adjusting_crop = False
        self.current_camera = None
        self.adjust_start_pos = None
        
        print("🎥 LeRobot Crop Visualizer")
        print("=" * 50)
        print("control instructions:")
        print("  ESC: exit program")
        print("  SPACE: pause/continue")
        print("  R: reset crop parameters")
        print("  mouse drag: adjust the crop region")
        print("  scroll wheel: adjust the crop region size")
        print("=" * 50)
        
    def load_crop_params(self) -> Dict[str, Tuple[int, int, int, int]]:
        """load crop parameters from the configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # try multiple possible paths
            crop_params_dict = {}
            
            # path 1: processor.crop_params_dict
            if "processor" in config and "crop_params_dict" in config["processor"]:
                crop_params_dict = config["processor"]["crop_params_dict"]
            
            # path 2: env.processor.image_preprocessing.crop_params_dict
            elif ("env" in config and "processor" in config["env"] and 
                  "image_preprocessing" in config["env"]["processor"] and 
                  "crop_params_dict" in config["env"]["processor"]["image_preprocessing"]):
                crop_params_dict = config["env"]["processor"]["image_preprocessing"]["crop_params_dict"]
            
            # path 3: image_preprocessing.crop_params_dict (directly)
            elif "image_preprocessing" in config and "crop_params_dict" in config["image_preprocessing"]:
                crop_params_dict = config["image_preprocessing"]["crop_params_dict"]
            
            # path 4: directly in the root level
            elif "crop_params_dict" in config:
                crop_params_dict = config["crop_params_dict"]
            
            print("📋 loaded crop parameters:")
            for key, params in crop_params_dict.items():
                print(f"  {key}: {params}")
            
            return crop_params_dict
        except Exception as e:
            print(f"❌ failed to load configuration file: {e}")
            return {}
    
    def save_crop_params(self):
        """save crop parameters to the configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # determine the save location - save to env.processor.image_preprocessing first
            if ("env" in config and "processor" in config["env"] and 
                "image_preprocessing" in config["env"]["processor"]):
                config["env"]["processor"]["image_preprocessing"]["crop_params_dict"] = self.crop_params
            elif "image_preprocessing" in config:
                config["image_preprocessing"]["crop_params_dict"] = self.crop_params
            elif "processor" in config:
                config["processor"]["crop_params_dict"] = self.crop_params
            else:
                # create env.processor.image_preprocessing part
                if "env" not in config:
                    config["env"] = {}
                if "processor" not in config["env"]:
                    config["env"]["processor"] = {}
                if "image_preprocessing" not in config["env"]["processor"]:
                    config["env"]["processor"]["image_preprocessing"] = {}
                config["env"]["processor"]["image_preprocessing"]["crop_params_dict"] = self.crop_params
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            print("💾 crop parameters have been saved to the configuration file")
        except Exception as e:
            print(f"❌ failed to save configuration file: {e}")
    
    def get_camera_device(self, camera_name: str) -> Optional[str]:
        """get camera device path"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            cameras = config.get("env", {}).get("robot", {}).get("cameras", {})
            camera_config = cameras.get(camera_name, {})
            return camera_config.get("index_or_path")
        except Exception as e:
            print(f"❌ failed to get camera configuration: {e}")
            return None
    
    def capture_camera(self, camera_name: str, device_path: str):
        """capture camera image"""
        print(f"📹 try to open camera {camera_name}: {device_path}")
        
        # try different device paths
        device_paths = [device_path]
        if "/dev/video" in device_path:
            # if /dev/video format, try different indices
            base_path = "/dev/video"
            for i in range(5):  # 尝试0-4
                alt_path = f"{base_path}{i}"
                if alt_path not in device_paths:
                    device_paths.append(alt_path)
        
        cap = None
        for path in device_paths:
            print(f"  🔍 try path: {path}")
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                print(f"  ✅ successfully opened: {path}")
                break
            else:
                print(f"  ❌ failed to open: {path}")
                cap.release()
        
        if not cap or not cap.isOpened():
            print(f"❌ failed to open any camera path {camera_name}")
            return
        
        # set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"📹 start capturing camera {camera_name}")
        frame_count = 0
        
        while self.running:
            if not self.paused:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    
                    # ensure image size is correct
                    if frame.shape[:2] != (480, 640):
                        frame = cv2.resize(frame, (640, 480))
                    
                    # convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.original_images[camera_name] = frame_rgb
                    
                    # apply crop
                    if camera_name in self.crop_params:
                        self.apply_crop(camera_name, frame_rgb)
                    
                    if frame_count % 30 == 0:  # print every 30 frames
                        print(f"  📸 {camera_name}: captured {frame_count} frames")
                else:
                    print(f"  ⚠️ {camera_name}: failed to read frame")
            
            time.sleep(1/30)  # 30 FPS
        
        cap.release()
        print(f"📹 camera {camera_name} has been closed")
    
    def apply_crop(self, camera_name: str, image: np.ndarray):
        """apply crop to the image"""
        if camera_name not in self.crop_params:
            return
        
        try:
            # convert to tensor (H, W, C) -> (C, H, W)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            # get crop parameters
            crop_params = self.crop_params[camera_name]
            
            # apply crop: F.crop(tensor, top, left, height, width)
            cropped_tensor = F.crop(image_tensor, *crop_params)
            
            # convert back to numpy (C, H, W) -> (H, W, C)
            cropped_image = (cropped_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            self.cropped_images[camera_name] = cropped_image
        except Exception as e:
            print(f"❌ failed to apply crop {camera_name}: {e}")
    
    def draw_crop_overlay(self, image: np.ndarray, camera_name: str) -> np.ndarray:
        """draw crop region on the image"""
        if camera_name not in self.crop_params:
            return image
        
        overlay = image.copy()
        crop_params = self.crop_params[camera_name]
        top, left, height, width = crop_params
        
        # draw crop region
        cv2.rectangle(overlay, (left, top), (left + width, top + height), (0, 255, 0), 2)
        
        # draw crop parameters text
        text = f"Crop: ({top}, {left}, {height}, {width})"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return overlay
    
    def mouse_callback(self, event, x, y, flags, param):
        """mouse callback function"""
        camera_name = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.adjusting_crop = True
            self.current_camera = camera_name
            self.adjust_start_pos = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.adjusting_crop and self.current_camera == camera_name:
            if camera_name in self.crop_params:
                top, left, height, width = self.crop_params[camera_name]
                
                # calculate new crop parameters
                dx = x - self.adjust_start_pos[0]
                dy = y - self.adjust_start_pos[1]
                
                new_left = max(0, min(left + dx, 640 - width))
                new_top = max(0, min(top + dy, 480 - height))
                
                self.crop_params[camera_name] = (new_top, new_left, height, width)
                self.adjust_start_pos = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.adjusting_crop = False
            self.current_camera = None
            self.adjust_start_pos = None
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            if camera_name in self.crop_params:
                top, left, height, width = self.crop_params[camera_name]
                
                # scroll wheel adjust size
                scale = 1.1 if flags > 0 else 0.9
                new_width = max(50, min(int(width * scale), 640 - left))
                new_height = max(50, min(int(height * scale), 480 - top))
                
                self.crop_params[camera_name] = (top, left, new_height, new_width)
    
    def start_cameras(self):
        """start all cameras"""
        # extract camera names from crop parameters
        camera_mapping = {}
        
        for crop_key in self.crop_params.keys():
            if "webcam_1" in crop_key:
                camera_mapping[crop_key] = "/dev/video0"
            elif "webcam_2" in crop_key:
                camera_mapping[crop_key] = "/dev/video2"  # according to the configuration file use video2
        
        print(f"📋 camera mapping:")
        for crop_key, device_path in camera_mapping.items():
            print(f"  {crop_key} -> {device_path}")
        
        for camera_name in self.crop_params.keys():
            device_path = self.get_camera_device(camera_name) or camera_mapping.get(camera_name)
            if device_path:
                print(f"🚀 start camera thread: {camera_name}")
                thread = threading.Thread(
                    target=self.capture_camera, 
                    args=(camera_name, device_path),
                    daemon=True
                )
                thread.start()
                self.camera_threads[camera_name] = thread
                time.sleep(1)  # increase startup interval
    
    def run(self):
        """run the visualizer"""
        # start cameras
        self.start_cameras()
        time.sleep(2)  # wait for cameras to start
        
        # create windows
        window_names = []
        for camera_name in self.crop_params.keys():
            window_name = f"LeRobot Crop Visualizer - {camera_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, self.mouse_callback, camera_name)
            window_names.append((camera_name, window_name))
        
        print("🚀 visualizer has been started, waiting for camera images...")
        
        # check camera status
        print("📊 camera status:")
        for camera_name in self.crop_params.keys():
            if camera_name in self.original_images:
                print(f"  ✅ {camera_name}: connected")
            else:
                print(f"  ❌ {camera_name}: not connected")
        
        try:
            while self.running:
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE
                    self.paused = not self.paused
                    print(f"{'⏸️ pause' if self.paused else '▶️ continue'}")
                elif key == ord('r') or key == ord('R'):  # R
                    self.crop_params = self.load_crop_params()
                    print("🔄 crop parameters have been reset")
                elif key == ord('s') or key == ord('S'):  # S
                    self.save_crop_params()
                
                # display images
                has_images = False
                for camera_name, window_name in window_names:
                    if camera_name in self.original_images and self.original_images[camera_name] is not None:
                        has_images = True
                        
                        # display original image with crop overlay
                        original_with_overlay = self.draw_crop_overlay(
                            self.original_images[camera_name], camera_name
                        )
                        
                        # ensure image is BGR format for display
                        display_image = cv2.cvtColor(original_with_overlay, cv2.COLOR_RGB2BGR)
                        cv2.imshow(window_name, display_image)
                        
                        # display cropped image
                        cropped_window_name = f"Cropped - {camera_name}"
                        if camera_name in self.cropped_images and self.cropped_images[camera_name] is not None:
                            if cropped_window_name not in [wn[1] for wn in window_names]:
                                cv2.namedWindow(cropped_window_name, cv2.WINDOW_NORMAL)
                            
                            cropped_display = cv2.cvtColor(self.cropped_images[camera_name], cv2.COLOR_RGB2BGR)
                            cv2.imshow(cropped_window_name, cropped_display)
                
                # if no images, display waiting information
                if not has_images:
                    # create waiting image
                    wait_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(wait_image, "Waiting for camera...", (200, 240), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    for camera_name, window_name in window_names:
                        cv2.imshow(window_name, wait_image)
        
        except KeyboardInterrupt:
            print("\n⏹️ user interrupted")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """cleanup resources"""
        print("🧹 cleanup resources...")
        self.running = False
        
        # wait for camera threads to end
        for thread in self.camera_threads.values():
            thread.join(timeout=1)
        
        cv2.destroyAllWindows()
        print("✅ cleanup completed")

def create_test_image(height=480, width=640):
    """create test image"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # add colored background
    for i in range(height):
        for j in range(width):
            image[i, j] = [
                int(255 * i / height),  # R channel
                int(255 * j / width),   # G channel
                int(255 * (i + j) / (height + width))  # B channel
            ]
    
    # add some geometric shapes
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)  # white rectangle
    cv2.circle(image, (320, 240), 50, (255, 0, 0), -1)  # blue circle
    cv2.putText(image, "TEST IMAGE", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image

def main():
    parser = argparse.ArgumentParser(description="LeRobot Crop Visualizer")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/ufactory/env_config_hilserl_lite6.json",
        help="configuration file path"
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="use test image mode (when cameras are not available)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.config_path).exists():
        print(f"❌ configuration file does not exist: {args.config_path}")
        return
    
    visualizer = CropVisualizer(args.config_path)
    
    if not visualizer.crop_params:
        print("❌ failed to find crop parameters, please check the configuration file")
        return
    
    # if test mode, use test image
    if args.test_mode:
        print("🧪 test mode: use test image")
        for camera_name in visualizer.crop_params.keys():
            test_image = create_test_image()
            visualizer.original_images[camera_name] = test_image
            visualizer.apply_crop(camera_name, test_image)
    
    try:
        visualizer.run()
    except Exception as e:
        print(f"❌ runtime error: {e}")
    finally:
        visualizer.cleanup()

if __name__ == "__main__":
    main()
