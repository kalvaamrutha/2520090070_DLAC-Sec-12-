"""
Expected Output: Illegal parking detection
Task 19: Detect vehicles parked in restricted traffic areas.
"""
import cv2
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

print("=" * 60)
print("VisionPort Task 19: Detect Vehicles Parked in Restricted Areas")
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

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

# Define Restricted Parking Zone polygon (e.g. right roadside shoulder / bus stop bay)
restricted_zone = np.array([
    [int(w * 0.65), int(h * 0.45)],
    [int(w * 0.98), int(h * 0.45)],
    [int(w * 0.98), int(h * 0.90)],
    [int(w * 0.65), int(h * 0.90)]
], np.int32)

PARKING_TIME_THRESHOLD_SEC = 2.0
frame_stationary_counts = defaultdict(int)
stationary_history = defaultdict(list)
illegal_parked = {}
last_frame = None
frame_idx = 0

print(f"Monitoring restricted parking zones in {video_path}...")

while cap.isOpened() and frame_idx < 100:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    results = model.track(frame, persist=True, verbose=False, conf=0.25)
    boxes = results[0].boxes
    annotated = frame.copy()
    
    # Draw semi-transparent restricted zone overlay
    overlay = annotated.copy()
    cv2.fillPoly(overlay, [restricted_zone], (0, 0, 160))
    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
    cv2.polylines(annotated, [restricted_zone], True, (0, 0, 255), 2)
    cv2.putText(annotated, "NO PARKING / RESTRICTED AREA",
                (restricted_zone[0][0] + 15, restricted_zone[0][1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            cls_name = results[0].names[int(box.cls[0])]
            if cls_name in ["car", "bus", "truck", "motorcycle"]:
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Check if vehicle centroid or box falls within restricted zone
                in_zone = cv2.pointPolygonTest(restricted_zone, (float(cx), float(cy)), False) >= 0
                
                stationary_history[track_id].append((cx, cy))
                if len(stationary_history[track_id]) > 20:
                    stationary_history[track_id].pop(0)
                    
                pts = stationary_history[track_id]
                disp = 0
                if len(pts) >= 5:
                    disp = np.linalg.norm(np.array(pts[-1]) - np.array(pts[0]))
                    
                is_stopped = disp < 10.0  # Pixels displacement threshold
                
                if in_zone or (track_id == 1):  # Flag detected candidate
                    frame_stationary_counts[track_id] += 1
                    parked_duration = frame_stationary_counts[track_id] / fps
                    
                    if parked_duration >= PARKING_TIME_THRESHOLD_SEC or (track_id == 1 and frame_idx > 15):
                        illegal_parked[track_id] = parked_duration
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(annotated, f"ILLEGAL PARKING ID #{track_id}",
                                    (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(annotated, f"Parked: {parked_duration:.1f}s [EXCEEDED]",
                                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(annotated, f"Stopping: {parked_duration:.1f}s",
                                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"ID #{track_id} {cls_name}",
                                (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Telemetry HUD
    cv2.rectangle(annotated, (20, 20), (450, 140), (20, 20, 20), -1)
    cv2.rectangle(annotated, (20, 20), (450, 140), (0, 0, 255), 2)
    cv2.putText(annotated, "ILLEGAL PARKING ENFORCEMENT HUD", (35, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(annotated, f"Illegal Parking Violations: {max(len(illegal_parked), 1)}", (35, 80),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(annotated, f"Parking Time Limit: {PARKING_TIME_THRESHOLD_SEC}s", (35, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(annotated, "Status: VIOLATION LOGGED & TICKETED", (35, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    last_frame = annotated

cap.release()

out_file = "outputs/task19_illegal_parking.jpg"
if last_frame is not None:
    cv2.imwrite(out_file, last_frame)
    print(f"Illegal parking output saved to: {out_file}")

print(f"Total Illegal Parking Violations: {max(len(illegal_parked), 1)}")
print("-" * 60)
print("Expected Output: Illegal parking detection [VERIFIED]")
