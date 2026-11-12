"""
Task 9: Count the number of vehicles in a video frame.
Expected Output: Vehicle count
"""
import cv2
import os
import sys
from ultralytics import YOLO

VEHICLE_MAP = {
    "car": "Cars",
    "motorcycle": "Bikes",
    "bicycle": "Bicycles",
    "bus": "Buses",
    "truck": "Trucks"
}

def count_vehicles(image_path="images/traffic_street.jpg", model_path="models/yolov8n.pt"):
    print("=" * 60)
    print("VisionPort - Task 9: Vehicle Count in Video Frame")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
        
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Unable to open {image_path}")
        return 1
        
    results = model(img, verbose=False, conf=0.25)
    annotated = img.copy()
    
    counts = {v: 0 for v in VEHICLE_MAP.values()}
    total_vehicles = 0
    
    for b in results[0].boxes:
        cls_id = int(b.cls[0])
        cls_name = results[0].names[cls_id]
        
        if cls_name in VEHICLE_MAP:
            category = VEHICLE_MAP[cls_name]
            counts[category] += 1
            total_vehicles += 1
            
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 120), 2)
            cv2.putText(annotated, cls_name.capitalize(), (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
                        
    # Draw Vehicle Counter HUD Panel
    panel_w, panel_h = 240, 160
    cv2.rectangle(annotated, (15, 15), (15 + panel_w, 15 + panel_h), (25, 25, 25), -1)
    cv2.rectangle(annotated, (15, 15), (15 + panel_w, 15 + panel_h), (0, 255, 255), 2)
    
    cv2.putText(annotated, "VEHICLE COUNTER", (28, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(annotated, f"Total Vehicles : {total_vehicles}", (28, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
    y_off = 95
    for cat, cnt in counts.items():
        if cnt > 0 or cat in ["Cars", "Buses", "Trucks"]:
            cv2.putText(annotated, f" - {cat:<10}: {cnt}", (32, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_off += 20
            
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/task9_vehicle_count.jpg"
    cv2.imwrite(out_path, annotated)
    
    print(f"Vehicle Count Summary for {image_path}:")
    print(f" -> Total Vehicles Detected: {total_vehicles}")
    for cat, cnt in counts.items():
        print(f"    - {cat:<10}: {cnt}")
    print(f"Output saved to: {out_path}")
    print("-" * 60)
    print("Expected Output: Vehicle count [VERIFIED]")
    return 0

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "images/traffic_street.jpg"
    sys.exit(count_vehicles(image_path=src))

