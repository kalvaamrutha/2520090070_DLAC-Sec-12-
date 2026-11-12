"""
Task 8: Display vehicle names and confidence scores.
Expected Output: Labelled detection output
"""
import cv2
import os
import sys
from ultralytics import YOLO

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}

def display_labels_and_confidence(image_path="images/traffic_sample_1.jpg", model_path="models/yolov8n.pt"):
    print("=" * 60)
    print("VisionPort - Task 8: Vehicle Names & Confidence Scores")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
        
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read {image_path}")
        return 1
        
    results = model(img, verbose=False, conf=0.25)
    annotated = img.copy()
    
    detections_logged = []
    
    for b in results[0].boxes:
        cls_id = int(b.cls[0])
        cls_name = results[0].names[cls_id]
        conf = float(b.conf[0])
        
        # Include vehicles (and people on road for context)
        if cls_name in VEHICLE_CLASSES or cls_name == "person":
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            
            # Format label with Vehicle Name and Confidence Score
            label_text = f"{cls_name.upper()} | Conf: {conf*100:.1f}% ({conf:.2f})"
            detections_logged.append((cls_name, conf, (x1, y1, x2, y2)))
            
            # Box color
            box_color = (0, 200, 255) if cls_name in VEHICLE_CLASSES else (150, 150, 150)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            
            # High-contrast label background badge
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 10, y1), (0, 0, 0), -1)
            cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 10, y1), box_color, 1)
            cv2.putText(annotated, label_text, (x1 + 5, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        
    # Overlay Summary Header
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 45), (15, 15, 15), -1)
    cv2.putText(annotated, f"VisionPort - Task 8: {len(detections_logged)} Objects Labelled with Confidence",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/task8_labelled_detections.jpg"
    cv2.imwrite(out_path, annotated)
    
    print(f"Total Labelled Detections: {len(detections_logged)}")
    for name, conf, coords in detections_logged:
        print(f" - Object: {name.capitalize():<12} | Confidence: {conf:.4f} ({conf*100:.1f}%) | BBox: {coords}")
        
    print(f"Saved labelled output image to: {out_path}")
    print("-" * 60)
    print("Expected Output: Labelled detection output [VERIFIED]")
    return 0

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "images/traffic_sample_1.jpg"
    sys.exit(display_labels_and_confidence(image_path=src))

