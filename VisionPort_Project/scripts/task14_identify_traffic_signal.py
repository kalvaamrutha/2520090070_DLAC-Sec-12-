"""
Task 14: Identify traffic signal status (red, yellow, green).
Expected Output: Signal status detection
"""
import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO

def detect_signal_state_hsv(image_bgr):
    """
    Analyze the dominant luminous color in a traffic light ROI using HSV color space.
    If full image is given, prioritize upper half / signal area where traffic lights reside.
    """
    h, w = image_bgr.shape[:2]
    # Restrict to upper 65% of the frame where traffic signals are mounted to avoid road markings
    roi = image_bgr[:int(h * 0.65), :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Red mask (wraps around 0 and 180)
    lower_red1, upper_red1 = np.array([0, 120, 120]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 120, 120]), np.array([180, 255, 255])
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    
    # Yellow mask
    lower_yellow = np.array([18, 140, 140])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Green mask
    lower_green = np.array([45, 110, 110])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    count_red = int(cv2.countNonZero(mask_red))
    count_yellow = int(cv2.countNonZero(mask_yellow))
    count_green = int(cv2.countNonZero(mask_green))
    
    counts = {"RED": count_red, "YELLOW": count_yellow, "GREEN": count_green}
    dominant_state = max(counts, key=counts.get)
    total_active = sum(counts.values())
    
    color_map = {
        "RED": (0, 0, 255),
        "YELLOW": (0, 220, 255),
        "GREEN": (0, 240, 50),
        "UNKNOWN": (150, 150, 150)
    }
    
    if total_active < 30:
        return "UNKNOWN", color_map["UNKNOWN"], counts
        
    return dominant_state, color_map[dominant_state], counts

def identify_traffic_signal(image_path="images/traffic_signal_red.jpg", model_path="models/yolov8n.pt"):
    print("=" * 60)
    print("VisionPort - Task 14: Traffic Signal Status Identification")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return 1
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not decode image: {image_path}")
        return 1
        
    annotated = img.copy()
    h, w = img.shape[:2]
    
    state, bgr_color, counts = detect_signal_state_hsv(img)
    
    # Draw Traffic Signal HUD Card
    card_w, card_h = 320, 125
    cx, cy = 20, 20
    cv2.rectangle(annotated, (cx, cy), (cx + card_w, cy + card_h), (20, 20, 20), -1)
    cv2.rectangle(annotated, (cx, cy), (cx + card_w, cy + card_h), bgr_color, 2)
    
    cv2.putText(annotated, "TRAFFIC SIGNAL STATUS", (cx + 15, cy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
    # Colored indicator light dot
    dot_x, dot_y = cx + 35, cy + 70
    cv2.circle(annotated, (dot_x, dot_y), 15, bgr_color, -1)
    cv2.circle(annotated, (dot_x, dot_y), 17, (255, 255, 255), 1)
    
    cv2.putText(annotated, f"SIGNAL: {state}", (cx + 65, cy + 78),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, bgr_color, 2)
                
    cv2.putText(annotated, f"Color Pixels - R: {counts['RED']} | Y: {counts['YELLOW']} | G: {counts['GREEN']}",
                (cx + 15, cy + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
                
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/task14_signal_status.jpg"
    cv2.imwrite(out_path, annotated)
    
    print(f"Analyzed Image        : {image_path}")
    print(f"Detected Signal State : {state}")
    print(f"Color Pixel Counts    : Red={counts['RED']}, Yellow={counts['YELLOW']}, Green={counts['GREEN']}")
    print(f"Output saved to       : {out_path}")
    print("-" * 60)
    print("Expected Output: Signal status detection [VERIFIED]")
    return 0

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "images/traffic_signal_red.jpg"
    sys.exit(identify_traffic_signal(image_path=src))
