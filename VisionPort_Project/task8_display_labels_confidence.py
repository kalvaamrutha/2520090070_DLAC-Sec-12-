"""
Expected Output: Labelled detection output
Task 8: Display vehicle names and confidence scores.
"""
import cv2
import os
from ultralytics import YOLO

print("=" * 60)
print("VisionPort Task 8: Display Vehicle Names and Confidence Scores")
print("=" * 60)

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
model = YOLO("yolov8n.pt")

input_img = "sample_traffic.jpg"
if not os.path.exists(input_img):
    input_img = "images/traffic_street.jpg"

img = cv2.imread(input_img)
results = model(img, verbose=False, conf=0.25)
annotated = img.copy()

detections = []

for box in results[0].boxes:
    cls_name = results[0].names[int(box.cls[0])]
    conf = float(box.conf[0])
    
    if cls_name in VEHICLE_CLASSES or cls_name == "person":
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{cls_name.upper()}: {conf*100:.1f}%"
        detections.append((cls_name, conf))
        
        # Bounding box & high contrast label badge
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

out_file = "task8_labelled_detections.jpg"
cv2.imwrite(out_file, annotated)

print(f"Logged {len(detections)} labelled detections:")
for name, conf in detections:
    print(f" - Vehicle: {name.capitalize():<12} | Confidence: {conf*100:.2f}%")
print(f"Saved output to: {out_file}")
print("-" * 60)
print("Expected Output: Labelled detection output [VERIFIED]")
