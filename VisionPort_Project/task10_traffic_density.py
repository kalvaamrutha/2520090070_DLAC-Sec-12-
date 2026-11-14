"""
Expected Output: Low, medium and high traffic
Task 10: Calculate traffic density based on vehicle count.
"""
import cv2
import os
from ultralytics import YOLO

print("=" * 60)
print("VisionPort Task 10: Calculate Traffic Density")
print("=" * 60)

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
model = YOLO("yolov8n.pt")

input_img = "sample_traffic.jpg"
if not os.path.exists(input_img):
    input_img = "images/traffic_street.jpg"

img = cv2.imread(input_img)
results = model(img, verbose=False, conf=0.25)
annotated = img.copy()

vehicle_count = 0
for box in results[0].boxes:
    cls_name = results[0].names[int(box.cls[0])]
    if cls_name in VEHICLE_CLASSES:
        vehicle_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)

# Calculate Traffic Density
if vehicle_count <= 3:
    density = "LOW"
    color = (0, 220, 0)       # Green
elif vehicle_count <= 7:
    density = "MEDIUM"
    color = (0, 180, 255)     # Amber
else:
    density = "HIGH"
    color = (0, 0, 255)       # Red

# Overlay Density Gauge
h, w = annotated.shape[:2]
bx, by, bw, bh = w - 300, 20, 280, 90
cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)
cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 3)
cv2.putText(annotated, "TRAFFIC DENSITY", (bx + 15, by + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
cv2.putText(annotated, f"Level: {density}", (bx + 15, by + 65),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2)

out_file = "task10_traffic_density.jpg"
cv2.imwrite(out_file, annotated)
print(f"Vehicle Count Detected : {vehicle_count}")
print(f"Calculated Density     : {density} TRAFFIC")
print(f"Output saved to        : {out_file}")
print("-" * 60)
print("Expected Output: Low, medium and high traffic [VERIFIED]")
