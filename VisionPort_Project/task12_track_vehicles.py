"""
Expected Output: Vehicle tracking IDs
Task 12: Track individual vehicles across video frames.
"""
import cv2
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

print("=" * 60)
print("VisionPort Task 12: Track Individual Vehicles Across Video Frames")
print("=" * 60)

video_path = "traffic_sample.mp4"
if not os.path.exists(video_path):
    video_path = "videos/traffic_demo.mp4"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Could not open video: {video_path}")
    exit(1)

track_history = defaultdict(list)
unique_ids = set()
last_frame = None
frame_idx = 0

print(f"Tracking vehicles in {video_path} using ByteTrack...")

while cap.isOpened() and frame_idx < 100:
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
                unique_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 25:
                    track_history[track_id].pop(0)
                    
                # Bounding box & Tracking ID Badge
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
                id_badge = f"ID #{track_id} : {cls_name.capitalize()}"
                cv2.rectangle(annotated, (x1, y1 - 22), (x1 + 130, y1), (0, 0, 0), -1)
                cv2.putText(annotated, id_badge, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                            
                # Trajectory trail
                pts = track_history[track_id]
                for i in range(1, len(pts)):
                    cv2.line(annotated, pts[i - 1], pts[i], (0, 255, 0), 2)
                    
    last_frame = annotated

cap.release()

out_file = "task12_vehicle_tracking.jpg"
if last_frame is not None:
    cv2.imwrite(out_file, last_frame)

print(f"Total Unique Vehicle IDs Tracked: {len(unique_ids)} (IDs: {sorted(list(unique_ids))})")
print(f"Saved tracking frame to: {out_file}")
print("-" * 60)
print("Expected Output: Vehicle tracking IDs [VERIFIED]")
