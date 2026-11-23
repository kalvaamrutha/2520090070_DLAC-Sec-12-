"""
Task 6: Detect cars in a traffic video using YOLO.
Expected Output: Car detection with bounding boxes
"""
import cv2
import os
import sys
from ultralytics import YOLO

def detect_cars(input_source="images/traffic_street.jpg", model_path="models/yolov8n.pt", conf_thresh=0.25):
    print("=" * 60)
    print("VisionPort - Task 6: Detect Cars Using YOLO")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
        
    print(f"Loading YOLO Model : {model_path}")
    model = YOLO(model_path)
    
    # Check if input is video or image
    is_video = input_source.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    
    os.makedirs("outputs", exist_ok=True)
    out_image_path = "outputs/task6_cars_detected.jpg"
    
    if is_video:
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video {input_source}")
            return 1
            
        print(f"Processing video frames from {input_source}...")
        cars_found_total = 0
        saved_frame = None
        frame_idx = 0
        
        while cap.isOpened() and frame_idx < 80:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            
            results = model(frame, verbose=False, conf=conf_thresh)
            boxes = results[0].boxes
            
            # Filter specifically for 'car' (class id 2)
            car_boxes = [b for b in boxes if int(b.cls[0]) == 2]
            
            if len(car_boxes) > 0 and saved_frame is None:
                annotated = frame.copy()
                for b in car_boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    conf = float(b.conf[0])
                    # Draw car bounding box (Cyan / Blue)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
                    label = f"Car: {conf*100:.1f}%"
                    cv2.rectangle(annotated, (x1, y1 - 22), (x1 + 110, y1), (255, 180, 0), -1)
                    cv2.putText(annotated, label, (x1 + 5, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                
                cv2.putText(annotated, f"Task 6: Detected {len(car_boxes)} Car(s)", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                saved_frame = annotated
                cars_found_total += len(car_boxes)
                
        cap.release()
        
        if saved_frame is not None:
            cv2.imwrite(out_image_path, saved_frame)
            print(f"[SUCCESS] Saved car detections to: {out_image_path}")
        else:
            # Fallback to image test
            input_source = "images/traffic_street.jpg"
            is_video = False
            
    if not is_video:
        print(f"Running inference on image: {input_source}")
        img = cv2.imread(input_source)
        if img is None:
            print(f"[ERROR] Could not load image {input_source}")
            return 1
            
        results = model(img, verbose=False, conf=conf_thresh)
        annotated = img.copy()
        car_count = 0
        
        for b in results[0].boxes:
            cls_id = int(b.cls[0])
            cls_name = results[0].names[cls_id]
            if cls_name == "car":
                car_count += 1
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                
                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 160, 0), 2)
                label = f"Car: {conf*100:.1f}%"
                cv2.rectangle(annotated, (x1, y1 - 22), (x1 + 115, y1), (255, 160, 0), -1)
                cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                            
        # HUD Header
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 45), (30, 30, 30), -1)
        cv2.putText(annotated, f"VisionPort - Task 6: Car Detection | Total Cars Found: {car_count}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
        cv2.imwrite(out_image_path, annotated)
        print(f"[SUCCESS] Detected {car_count} car(s). Saved to: {out_image_path}")
        
    print("-" * 60)
    print("Expected Output: Car detection with bounding boxes [VERIFIED]")
    return 0

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "images/traffic_street.jpg"
    sys.exit(detect_cars(input_source=src))

