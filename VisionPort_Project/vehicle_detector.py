"""
Vehicle Detector Module for VisionPort Project
Provides reusable vehicle detection, classification, counting, and density analysis.
"""
import cv2
import numpy as np
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.25):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        # COCO vehicle classes: 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = {
            "car": (255, 180, 0),        # Cyan / Light Blue
            "motorcycle": (0, 220, 0),   # Green
            "bicycle": (50, 255, 150),   # Light Green
            "bus": (0, 140, 255),        # Orange
            "truck": (255, 0, 200)       # Magenta
        }

    def detect(self, frame):
        """
        Run detection and return:
        - annotated_frame
        - detections list: [{"class": name, "confidence": conf, "bbox": [x1, y1, x2, y2]}]
        - counts dict: {"car": N, "motorcycle": N, "bus": N, "truck": N}
        - total_vehicles int
        - density_str: "LOW" | "MEDIUM" | "HIGH"
        """
        results = self.model(frame, verbose=False, conf=self.conf_thresh)
        annotated = frame.copy()
        
        detections = []
        counts = {k: 0 for k in self.target_classes}
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Map class name
            v_type = None
            if cls_name in self.target_classes:
                v_type = cls_name
            elif "bike" in cls_name:
                v_type = "motorcycle"
                
            if v_type:
                counts[v_type] += 1
                color = self.target_classes[v_type]
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"{v_type.upper()} {conf*100:.1f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                            
                detections.append({
                    "class": v_type,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })
                
        total_vehicles = sum(counts.values())
        
        # Density classification
        if total_vehicles <= 3:
            density = "LOW"
        elif total_vehicles <= 7:
            density = "MEDIUM"
        else:
            density = "HIGH"
            
        return annotated, detections, counts, total_vehicles, density

if __name__ == "__main__":
    detector = VehicleDetector()
    img = cv2.imread("sample_traffic.jpg")
    if img is not None:
        annotated, dets, counts, total, density = detector.detect(img)
        print(f"Detected {total} vehicles: {counts} | Density: {density}")
        cv2.imwrite("output_test.jpg", annotated)
        print("Output saved to output_test.jpg")
    else:
        print("sample_traffic.jpg not found.")
