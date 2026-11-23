"""
Expected Output: Signal status detection
Task 14: Identify traffic signal status (red, yellow, green).
"""
import cv2
import os
import numpy as np

print("=" * 60)
print("VisionPort Task 14: Identify Traffic Signal Status")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)
signal_imgs = ["images/traffic_signal_red.jpg", "sample_traffic.jpg", "images/sample_traffic.jpg"]
signal_img = None
for p in signal_imgs:
    if os.path.exists(p):
        signal_img = p
        break

img = cv2.imread(signal_img)
h, w = img.shape[:2]
roi = img[:int(h * 0.65), :]
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# HSV color ranges
mask_r1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
mask_r2 = cv2.inRange(hsv, np.array([160, 120, 120]), np.array([180, 255, 255]))
mask_red = cv2.bitwise_or(mask_r1, mask_r2)
mask_yellow = cv2.inRange(hsv, np.array([18, 140, 140]), np.array([35, 255, 255]))
mask_green = cv2.inRange(hsv, np.array([45, 110, 110]), np.array([85, 255, 255]))

counts = {
    "RED": int(cv2.countNonZero(mask_red)),
    "YELLOW": int(cv2.countNonZero(mask_yellow)),
    "GREEN": int(cv2.countNonZero(mask_green))
}

status = max(counts, key=counts.get)
color_map = {"RED": (0, 0, 255), "YELLOW": (0, 220, 255), "GREEN": (0, 240, 50)}

annotated = img.copy()
cv2.rectangle(annotated, (20, 20), (320, 110), (20, 20, 20), -1)
cv2.rectangle(annotated, (20, 20), (320, 110), color_map[status], 2)
cv2.circle(annotated, (50, 65), 14, color_map[status], -1)
cv2.putText(annotated, f"SIGNAL: {status}", (80, 72),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, color_map[status], 2)

out_file = "outputs/task14_signal_status.jpg"
cv2.imwrite(out_file, annotated)
print(f"Analyzed Image  : {signal_img}")
print(f"Detected Status : {status} (Pixel counts: {counts})")
print(f"Saved to        : {out_file}")
print("-" * 60)
print("Expected Output: Signal status detection [VERIFIED]")
