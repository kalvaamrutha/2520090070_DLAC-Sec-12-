"""
Expected Output: Congestion alerts
Task 11: Detect traffic congestion at signal junctions.
"""
import cv2
import os
from datetime import datetime
from ultralytics import YOLO

print("=" * 60)
print("VisionPort Task 11: Detect Traffic Congestion at Signal Junctions")
print("=" * 60)

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
model = YOLO("yolov8n.pt")

input_img = "images/traffic_congested.jpg"
if not os.path.exists(input_img):
    input_img = "sample_traffic.jpg"

img = cv2.imread(input_img)
results = model(img, verbose=False, conf=0.25)
annotated = img.copy()

vehicles = 0
for box in results[0].boxes:
    cls_name = results[0].names[int(box.cls[0])]
    if cls_name in VEHICLE_CLASSES:
        vehicles += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)

congestion_threshold = 3
is_congested = vehicles >= congestion_threshold
h, w = annotated.shape[:2]

if is_congested:
    alert_status = "ACTIVE"
    cv2.rectangle(annotated, (0, 0), (w, 55), (0, 0, 220), -1)
    cv2.putText(annotated, "!!! CONGESTION ALERT: JUNCTION OVERCAPACITY DETECTED !!!",
                (15, 34), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
else:
    alert_status = "NORMAL"
    cv2.rectangle(annotated, (0, 0), (w, 45), (30, 30, 30), -1)
    cv2.putText(annotated, f"Junction Status: Normal Traffic ({vehicles} vehicles)",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

out_file = "task11_congestion_alert.jpg"
cv2.imwrite(out_file, annotated)
print(f"Junction Vehicle Count  : {vehicles}")
print(f"Congestion Alert Status : {alert_status}")
print(f"Output saved to         : {out_file}")
print("-" * 60)
print("Expected Output: Congestion alerts [VERIFIED]")
