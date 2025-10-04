'''bash
python test/visualize_cams.py /home/zekaijin/lerobot-hilserl/configs/ufactory/env_config_hilserl_lite6.json
'''
import cv2, json, sys

cfg_path = sys.argv[1]
cfg = json.load(open(cfg_path))
cams = cfg["env"]["robot"]["cameras"]
crops = cfg["env"]["processor"]["image_preprocessing"]["crop_params_dict"]
resize = cfg["env"]["processor"]["image_preprocessing"]["resize_size"]

def open_cam(cam_cfg):
    src = cam_cfg["index_or_path"]
    cap = cv2.VideoCapture(src if isinstance(src, int) else src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])
    return cap

caps = {k: open_cam(v) for k, v in cams.items()}

while True:
    for name, cap in caps.items():
        ok, frame = cap.read()
        if not ok: continue

        # get ROI (config is [x,y,w,h])
        x, y, w, h = crops.get(f"observation.images.{name}", [0, 0, frame.shape[1], frame.shape[0]])
        x2, y2 = x + w, y + h

        # draw box on original image
        orig = frame.copy()
        cv2.rectangle(orig, (x, y), (x2, y2), (0, 255, 0), 2)

        # crop+resize
        roi = frame[y:y2, x:x2]
        if roi.size > 0:
            roi_resized = cv2.resize(roi, (resize[1], resize[0]))  # (w,h)
            # height align and display side by side
            hmin = min(orig.shape[0], roi_resized.shape[0])
            orig_show = cv2.resize(orig, (int(orig.shape[1]*hmin/orig.shape[0]), hmin))
            roi_show = cv2.resize(roi_resized, (int(roi_resized.shape[1]*hmin/roi_resized.shape[0]), hmin))
            show = cv2.hconcat([orig_show, roi_show])
        else:
            show = orig

        cv2.imshow(name, show)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
        break

for cap in caps.values(): cap.release()
cv2.destroyAllWindows()