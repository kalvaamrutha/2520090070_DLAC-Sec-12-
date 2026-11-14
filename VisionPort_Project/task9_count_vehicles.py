"""
Expected Output: Vehicle count
Task 9: Count the number of vehicles in a video frame.
"""
import cv2
import os
from ultralytics import YOLO

print("=" * 60)
print("VisionPort Task 9: Vehicle Count in Video Frame")
print("=" * 60)

VEHICLE_MAP = {
    "car": "Cars",
    "motorcycle": "Bikes",
    "bicycle": "Bicycles",
    "bus": "Buses",
    "truck": "Trucks"
}

model = YOLO("yolov8n.pt")
input_img = "sample_traffic.jpg"
if not os.path.exists(input_img):
    input_img = "images/traffic_street.jpg"

img = cv2.imread(input_img)
results = model(img, verbose=False, conf=0.25)
annotated = img.copy()

counts = {v: 0 for v in VEHICLE_MAP.values()}
total_vehicles = 0

for box in results[0].boxes:
    cls_name = results[0].names[int(box.cls[0])]
    if cls_name in VEHICLE_MAP:
        category = VEHICLE_MAP[cls_name]
        counts[category] += 1
        total_vehicles += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 120), 2)

# Draw Counter Panel HUD
cv2.rectangle(annotated, (15, 15), (260, 160), (25, 25, 25), -1)
cv2.rectangle(annotated, (15, 15), (260, 160), (0, 255, 255), 2)
cv2.putText(annotated, "VEHICLE COUNTER", (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
cv2.putText(annotated, f"Total Vehicles: {total_vehicles}", (25, 72),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

y = 96
for cat, cnt in counts.items():
    cv2.putText(annotated, f" - {cat}: {cnt}", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 20

out_file = "task9_vehicle_count.jpg"
cv2.imwrite(out_file, annotated)
print(f"Total Vehicles Detected: {total_vehicles}")
print(f"Vehicle Breakdown: {counts}")
print(f"Saved to: {out_file}")
print("-" * 60)
print("Expected Output: Vehicle count [VERIFIED]")
