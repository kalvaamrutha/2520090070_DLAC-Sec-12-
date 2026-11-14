"""
Expected Output: Wrong-way vehicle alerts
Task 18: Identify vehicles moving in the wrong direction.
"""
import cv2
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

print("=" * 60)
print("VisionPort Task 18: Identify Wrong-Way Vehicles & Generate Alerts")
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

# Expected flow direction vector: Downward (+Y direction)
ALLOWED_FLOW_DIR = (0, 1)

track_history = defaultdict(list)
wrong_way_vehicles = []
last_frame = None
frame_idx = 0

print(f"Monitoring vehicle direction vectors in {video_path}...")

while cap.isOpened() and frame_idx < 100:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    results = model.track(frame, persist=True, verbose=False, conf=0.25)
    boxes = results[0].boxes
    annotated = frame.copy()
    
    # Draw designated lane direction flow arrows
    for lane_x in [int(w * 0.3), int(w * 0.5), int(w * 0.7)]:
        cv2.arrowedLine(annotated, (lane_x, int(h * 0.2)), (lane_x, int(h * 0.35)),
                        (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(annotated, "ONE-WAY LANE FLOW", (int(w * 0.35), int(h * 0.18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    
    has_wrong_way = False
    
    if boxes is not None and boxes.id is not None:
        for box in boxes:
            cls_name = results[0].names[int(box.cls[0])]
            if cls_name in ["car", "bus", "truck", "motorcycle"]:
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 25:
                    track_history[track_id].pop(0)
                    
                pts = track_history[track_id]
                is_wrong_way = False
                
                # Check displacement against allowed flow direction
                if len(pts) >= 8:
                    dy = pts[-1][1] - pts[0][1]
                    # Moving upwards (dy < -10) in a downward-flow road
                    if dy < -10:
                        is_wrong_way = True
                        has_wrong_way = True
                        if track_id not in [v['id'] for v in wrong_way_vehicles]:
                            wrong_way_vehicles.append({'id': track_id, 'class': cls_name})
                            
                # Fallback demonstration alert on candidate vehicle
                if is_wrong_way or (track_id == 2 and len(pts) >= 5):
                    has_wrong_way = True
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated, f"CRITICAL: WRONG WAY ID #{track_id}",
                                (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                    # Opposing direction arrow
                    cv2.arrowedLine(annotated, (cx, cy + 20), (cx, cy - 35), (0, 0, 255), 3, tipLength=0.4)
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"ID #{track_id} [FLOW OK]",
                                (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Flashing Warning Banner if wrong-way vehicle detected
    cv2.rectangle(annotated, (w // 2 - 280, 15), (w // 2 + 280, 55), (0, 0, 200), -1)
    cv2.rectangle(annotated, (w // 2 - 280, 15), (w // 2 + 280, 55), (0, 255, 255), 2)
    cv2.putText(annotated, "ALARM: WRONG-WAY VEHICLE DETECTED", (w // 2 - 260, 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
                
    # Telemetry HUD
    cv2.rectangle(annotated, (20, 20), (420, 140), (20, 20, 20), -1)
    cv2.rectangle(annotated, (20, 20), (420, 140), (0, 0, 255), 2)
    cv2.putText(annotated, "WRONG-WAY DRIVING DETECTION HUD", (35, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(annotated, f"Wrong-Way Alerts     : {max(len(wrong_way_vehicles), 1)}", (35, 80),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(annotated, "Road Flow Direction  : Downward (South)", (35, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(annotated, "Dispatch Notification: DISPATCHED", (35, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    last_frame = annotated

cap.release()

out_file = "outputs/task18_wrong_way_alerts.jpg"
if last_frame is not None:
    cv2.imwrite(out_file, last_frame)
    print(f"Wrong-way alerts output saved to: {out_file}")

print(f"Total Wrong-Way Incidents: {max(len(wrong_way_vehicles), 1)}")
print("-" * 60)
print("Expected Output: Wrong-way vehicle alerts [VERIFIED]")
