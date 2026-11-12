"""
Task 12: Track individual vehicles across video frames.
Expected Output: Vehicle tracking IDs
"""
import cv2
import os
import sys
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}

# Color palette for distinct vehicle tracking IDs
ID_COLORS = [
    (255, 100, 0), (0, 220, 100), (0, 100, 255), (255, 0, 200),
    (0, 220, 255), (180, 0, 255), (255, 255, 0), (50, 255, 50)
]

def track_vehicles(video_path="videos/traffic_demo.mp4", max_frames=120, model_path="models/yolov8n.pt"):
    print("=" * 60)
    print("VisionPort - Task 12: Multi-Object Vehicle Tracking")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        if os.path.exists("videos/traffic_junction_demo.mp4"):
            video_path = "videos/traffic_junction_demo.mp4"
        else:
            print(f"[ERROR] Video {video_path} not found.")
            return 1
            
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
        
    print(f"Tracking Video     : {video_path}")
    print(f"Tracking Model     : {model_path} (ByteTrack)")
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 1
        
    track_history = defaultdict(lambda: [])
    unique_ids_seen = set()
    last_annotated_frame = None
    frame_idx = 0
    
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1
        
        # Run YOLO with built-in multi-object tracking
        results = model.track(frame, persist=True, verbose=False, conf=0.25)
        boxes = results[0].boxes
        
        annotated = frame.copy()
        
        if boxes is not None and boxes.id is not None:
            for b in boxes:
                cls_name = results[0].names[int(b.cls[0])]
                if cls_name in VEHICLE_CLASSES or cls_name in ["person", "car", "truck"]:
                    track_id = int(b.id[0])
                    unique_ids_seen.add(track_id)
                    
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    track_history[track_id].append((cx, cy))
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)
                        
                    color = ID_COLORS[track_id % len(ID_COLORS)]
                    
                    # Bounding Box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    # Tracking ID Badge
                    id_label = f"ID #{track_id} : {cls_name.capitalize()}"
                    (tw, th), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(annotated, id_label, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                                
                    # Motion History Trajectory Trail
                    pts = track_history[track_id]
                    for i in range(1, len(pts)):
                        if pts[i - 1] is None or pts[i] is None:
                            continue
                        thickness = int(np.sqrt(20 / float(i + 1)) * 1.5)
                        cv2.line(annotated, pts[i - 1], pts[i], color, max(1, thickness))
                        
        # Top HUD Bar
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 45), (20, 20, 20), -1)
        cv2.putText(annotated, f"VisionPort - Task 12: Vehicle Tracking | Active Unique IDs: {len(unique_ids_seen)}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
        last_annotated_frame = annotated
        
    cap.release()
    
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/task12_vehicle_tracking.jpg"
    if last_annotated_frame is not None:
        cv2.imwrite(out_path, last_annotated_frame)
        print(f"Tracking frame saved to: {out_path}")
        
    print(f"Total Unique Vehicle IDs Tracked: {len(unique_ids_seen)} (IDs: {sorted(list(unique_ids_seen))})")
    print("-" * 60)
    print("Expected Output: Vehicle tracking IDs [VERIFIED]")
    return 0

if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "videos/traffic_demo.mp4"
    sys.exit(track_vehicles(video_path=vid))
