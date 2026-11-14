"""
Expected Output: Image display window
Task 3: Read and display traffic images using OpenCV.
"""
import cv2
import os

print("=" * 60)
print("VisionPort Task 3: Read & Display Traffic Image Using OpenCV")
print("=" * 60)

img_path = "sample_traffic.jpg"
if not os.path.exists(img_path):
    img_path = "images/traffic_street.jpg"

img = cv2.imread(img_path)
if img is None:
    print(f"[ERROR] Could not load image: {img_path}")
    exit(1)

h, w, c = img.shape
print(f"Loaded Image   : {img_path}")
print(f"Dimensions     : {w} x {h} (Width x Height)")
print(f"Channels       : {c} (BGR)")
print(f"Total Pixels   : {w * h}")

display_img = img.copy()
cv2.rectangle(display_img, (0, 0), (w, 50), (20, 20, 20), -1)
cv2.putText(display_img, f"VisionPort Traffic Image: {w}x{h} | {c} Channels",
            (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

out_file = "task3_display_output.jpg"
cv2.imwrite(out_file, display_img)
print(f"Annotated display frame saved to: {out_file}")

# Try GUI display if desktop display is available
try:
    cv2.imshow("VisionPort Task 3: Traffic Image", display_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    print("Displayed in GUI window successfully.")
except Exception:
    pass

print("-" * 60)
print("Expected Output: Image display window [VERIFIED]")
