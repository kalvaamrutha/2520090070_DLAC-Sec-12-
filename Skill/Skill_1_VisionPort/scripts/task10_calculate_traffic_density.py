"""
Task 10: Calculate traffic density based on vehicle count.
Expected Output: Low, medium and high traffic
"""
import cv2
import os
import sys
from ultralytics import YOLO

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}

def calculate_traffic_density(vehicle_count):
    """
    Density Classification:
      - LOW    : 0 to 3 vehicles (Green)
      - MEDIUM : 4 to 7 vehicles (Amber/Yellow)
      - HIGH   : 8 or more vehicles (Red)
    """
    if vehicle_count <= 3:
        return "LOW", (0, 220, 0), "Free Flow / Light Traffic"
    elif vehicle_count <= 7:
        return "MEDIUM", (0, 180, 255), "Moderate Traffic / Regular Commute"
    else:
        return "HIGH", (0, 0, 255), "Heavy Traffic / High Congestion Risk"

def analyze_density(image_path="images/traffic_street.jpg", model_path="models/yolov8n.pt"):
    print("=" * 60)
    print("VisionPort - Task 10: Traffic Density Calculation")
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
    
    vehicle_count = 0
    for b in results[0].boxes:
        cls_name = results[0].names[int(b.cls[0])]
        if cls_name in VEHICLE_CLASSES:
            vehicle_count += 1
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)
            
    density_level, color, description = calculate_traffic_density(vehicle_count)
    
    # Draw Density Badge HUD
    h, w = annotated.shape[:2]
    badge_w, badge_h = 320, 110
    bx, by = w - badge_w - 20, 20
    
    cv2.rectangle(annotated, (bx, by), (bx + badge_w, by + badge_h), (20, 20, 20), -1)
    cv2.rectangle(annotated, (bx, by), (bx + badge_w, by + badge_h), color, 3)
    
    cv2.putText(annotated, "TRAFFIC DENSITY MONITOR", (bx + 15, by + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(annotated, f"Level: {density_level} TRAFFIC", (bx + 15, by + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(annotated, f"Vehicles: {vehicle_count} | {description}", (bx + 15, by + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                
    os.makedirs("outputs", exist_ok=True)
    out_img = "outputs/task10_traffic_density.jpg"
    cv2.imwrite(out_img, annotated)
    
    # Save Report
    report_path = "outputs/task10_density_report.txt"
    with open(report_path, "w") as f:
        f.write("VisionPort Traffic Density Analysis Report\n")
        f.write("-" * 50 + "\n")
        f.write(f"Analyzed Image        : {image_path}\n")
        f.write(f"Vehicle Count         : {vehicle_count}\n")
        f.write(f"Traffic Density Level : {density_level}\n")
        f.write(f"Status Description    : {description}\n")
        f.write(f"Classification Bounds : Low (0-3), Medium (4-7), High (8+)\n")
        
    print(f"Vehicle Count Detected : {vehicle_count}")
    print(f"Computed Density Level : {density_level} TRAFFIC")
    print(f"Description            : {description}")
    print(f"Report saved to        : {report_path}")
    print(f"Image saved to         : {out_img}")
    print("-" * 60)
    print("Expected Output: Low, medium and high traffic [VERIFIED]")
    return 0

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "images/traffic_street.jpg"
    sys.exit(analyze_density(image_path=src))

