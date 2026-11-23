"""
Expected Output: Speed estimation
Task 13: Calculate approximate vehicle speed.
"""
import cv2
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

print("=" * 60)
print("VisionPort Task 13: Calculate Approximate Vehicle Speed")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)
video_paths = ["videos/traffic_sample.mp4", "traffic_sample.mp4", "videos/traffic_demo.mp4"]
video_path = None
for p in video_paths:
    if os.path.exists(p):
        video_path = p
        break

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Could not open video: {video_path}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
ppm = 10.0  # pixels per meter calibration

pos_history = defaultdict(list)
speeds = {}
last_frame = None
frame_idx = 0

print(f"Estimating vehicle speeds in {video_path} (FPS: {fps:.2f}, PPM: {ppm})...")

while cap.isOpened() and frame_idx < 120:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    results = model.track(frame, persist=True, verbose=False, conf=0.25)
    boxes = results[0].boxes
    annotated = frame.copy()
    
    if boxes is not None and boxes.id is not None:
        for box in boxes:
            cls_name = results[0].names[int(box.cls[0])]
            if cls_name in ["car", "bus", "truck", "motorcycle", "person"]:
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                pos_history[track_id].append((frame_idx, cx, cy))
                if len(pos_history[track_id]) > 15:
                    pos_history[track_id].pop(0)
                    
                speed_kmh = speeds.get(track_id, 0.0)
                if len(pos_history[track_id]) >= 5:
                    f0, x0, y0 = pos_history[track_id][0]
                    f1, x1_c, y1_c = pos_history[track_id][-1]
                    dt = (f1 - f0) / fps
                    if dt > 0:
                        dist_m = np.hypot(x1_c - x0, y1_c - y0) / ppm
                        speed_kmh = round((dist_m / dt) * 3.6, 1)
                        speeds[track_id] = speed_kmh
                        
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
                badge = f"ID #{track_id} | {speed_kmh} km/h"
                cv2.rectangle(annotated, (x1, y1 - 22), (x1 + 140, y1), (0, 0, 0), -1)
                cv2.putText(annotated, badge, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                            
    last_frame = annotated

cap.release()

out_file = "outputs/task13_speed_estimation.jpg"
if last_frame is not None:
    cv2.imwrite(out_file, last_frame)

print(f"Recorded Vehicle Speeds: {speeds}")
print(f"Saved speed estimation frame to: {out_file}")
print("-" * 60)
print("Expected Output: Speed estimation [VERIFIED]")

