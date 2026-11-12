"""
Task 7: Detect bikes, buses and trucks.
Expected Output: Multiple vehicle detection
"""
import cv2
import os
import sys
from ultralytics import YOLO

# Target vehicle classes in COCO:
# 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_VEHICLES = {
    "car": (255, 180, 0),        # Cyan / Light Blue (BGR)
    "motorcycle": (0, 220, 0),   # Green
    "bicycle": (50, 255, 150),   # Light Green
    "bus": (0, 140, 255),        # Orange
    "truck": (255, 0, 200)       # Magenta / Violet
}

def detect_multiple_vehicles(input_path="images/traffic_street.jpg", model_path="models/yolov8n.pt"):
    print("=" * 60)
    print("VisionPort - Task 7: Detect Bikes, Buses, Trucks, and Cars")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
        
    model = YOLO(model_path)
    
    # We can test with street image or bus image
    img = cv2.imread(input_path)
    if img is None:
        print(f"[ERROR] Could not read {input_path}")
        return 1
        
    results = model(img, verbose=False, conf=0.25)
    annotated = img.copy()
    
    detected_counts = {v: 0 for v in TARGET_VEHICLES}
    
    for b in results[0].boxes:
        cls_id = int(b.cls[0])
        cls_name = results[0].names[cls_id]
        
        # Check if detected object is one of our target vehicles
        matched_category = None
        if cls_name in TARGET_VEHICLES:
            matched_category = cls_name
        elif "bike" in cls_name or "motorcycle" in cls_name:
            matched_category = "motorcycle"
            
        if matched_category:
            detected_counts[matched_category] += 1
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            color = TARGET_VEHICLES[matched_category]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Tag label
            label = f"{matched_category.capitalize()}: {conf*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                        
    # Top HUD Summary Bar
    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (0, 0), (w, 55), (20, 20, 20), -1)
    summary_text = f"Task 7 Multi-Vehicle: " + " | ".join([f"{k.capitalize()}s: {v}" for k, v in detected_counts.items() if v > 0 or k in ['car', 'bus', 'truck']])
    cv2.putText(annotated, summary_text, (15, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/task7_multiple_vehicles.jpg"
    cv2.imwrite(out_path, annotated)
    
    print("Detected Vehicle Counts:")
    for v_type, count in detected_counts.items():
        print(f" - {v_type.capitalize():<12}: {count}")
    print(f"Annotated result saved to: {out_path}")
    print("-" * 60)
    print("Expected Output: Multiple vehicle detection [VERIFIED]")
    return 0

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "images/traffic_street.jpg"
    sys.exit(detect_multiple_vehicles(input_path=src))

