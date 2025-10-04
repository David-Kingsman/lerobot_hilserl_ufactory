'''bash
python test/tune_roi.py /home/zekaijin/lerobot-hilserl/configs/ufactory/env_config_hilserl_lite6.json
'''
import json, sys, cv2

cfg_path = sys.argv[1]
cfg = json.load(open(cfg_path))
cams = cfg["env"]["robot"]["cameras"]
crops = cfg["env"]["processor"]["image_preprocessing"]["crop_params_dict"]
resize = cfg["env"]["processor"]["image_preprocessing"]["resize_size"]

def open_cam(c):
    cap = cv2.VideoCapture(c["index_or_path"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
    cap.set(cv2.CAP_PROP_FPS, c["fps"])
    return cap

caps = {k: open_cam(v) for k,v in cams.items()}
names = list(caps.keys())
cur = 0

def get_roi(name, frame):
    key = f"observation.images.{name}"
    # crop_params_dict格式: (top, left, height, width)
    top, left, height, width = crops.get(key, [0,0,frame.shape[0],frame.shape[1]])
    return [top, left, height, width], key

STEP = 5

while True:
    name = names[cur]
    ok, frame = caps[name].read()
    if not ok: continue
    roi, key = get_roi(name, frame)
    top, left, height, width = roi
    x, y = left, top
    x2, y2 = x + width, y + height
    show = frame.copy()
    cv2.rectangle(show, (x,y), (x2,y2), (0,255,0), 2)
    cv2.putText(show, f"{name} ROI (top,left,h,w)={[top,left,height,width]}", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("tune_roi", show)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')): break
    if k == ord('1'): cur = 0
    if k == ord('2') and len(names)>1: cur = 1
    if k == 81: left = max(0, left-STEP)           # left
    if k == 83: left = min(frame.shape[1]-width, left+STEP) # right
    if k == 82: top = max(0, top-STEP)           # up
    if k == 84: top = min(frame.shape[0]-height, top+STEP) # down
    if k == ord('['): width = max(10, width-STEP)
    if k == ord(']'): width = min(frame.shape[1]-left, width+STEP)
    if k == ord(';'): height = max(10, height-STEP)
    if k == ord("'"): height = min(frame.shape[0]-top, height+STEP)
    if k in (81,83,82,84,ord('['),ord(']'),ord(';'),ord("'")):
        crops[key] = [top, left, height, width]
    if k == ord('s'):
        cfg["env"]["processor"]["image_preprocessing"]["crop_params_dict"] = crops
        with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
        print("Saved:", crops)

for c in caps.values(): c.release()
cv2.destroyAllWindows()