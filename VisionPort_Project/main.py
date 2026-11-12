"""
VisionPort Live Portal - Enhanced FastAPI Backend
Integrating Deep Learning Vehicle Detection, Multi-Class Recognition,
Traffic Density Assessment, Congestion Alerts, Signal Identification,
and Queue Waiting-Time Estimation.
"""
import io
import time
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Import modular task logic
from scripts.task10_calculate_traffic_density import calculate_traffic_density
from scripts.task14_identify_traffic_signal import detect_signal_state_hsv
from scripts.task15_estimate_waiting_time import compute_waiting_time, CLEARANCE_WEIGHTS

# Initialize FastAPI Application
app = FastAPI(
    title="VisionPort - Intelligent Vehicle Detection & Traffic Monitoring API",
    description="Deep Learning API for Indian traffic analysis, density monitoring, and junction signal control.",
    version="2.0.0"
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and images
os.makedirs("static", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")

# Load YOLO Model
MODEL_PATH = "models/yolov8n.pt" if os.path.exists("models/yolov8n.pt") else "yolov8n.pt"
model = YOLO(MODEL_PATH)

# Vehicle classification mappings and colors
VEHICLE_COLORS = {
    "car": (255, 180, 0),        # Cyan / Light Blue
    "motorcycle": (0, 220, 0),   # Green
    "bicycle": (50, 255, 150),   # Light Green
    "bus": (0, 140, 255),        # Orange
    "truck": (255, 0, 200)       # Magenta
}

def analyze_traffic_frame(img_bgr, conf_thresh=0.25):
    """
    Core VisionPort Analysis Pipeline:
    1. YOLO Detection (Cars, Bikes, Buses, Trucks)
    2. Vehicle Counting & Breakdown
    3. Traffic Density Classification
    4. Junction Congestion Alert Check
    5. Traffic Signal Status Detection
    6. Waiting-Time Estimation
    7. Professional HUD Annotation
    """
    t_start = time.time()
    results = model(img_bgr, verbose=False, conf=conf_thresh)
    boxes = results[0].boxes
    
    annotated = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    
    vehicle_counts = {"car": 0, "motorcycle": 0, "bicycle": 0, "bus": 0, "truck": 0}
    detections_list = []
    
    for b in boxes:
        cls_id = int(b.cls[0])
        cls_name = results[0].names[cls_id]
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        
        # Categorize vehicle
        v_cat = None
        if cls_name in vehicle_counts:
            v_cat = cls_name
        elif "bike" in cls_name:
            v_cat = "motorcycle"
            
        if v_cat:
            vehicle_counts[v_cat] += 1
            color = VEHICLE_COLORS[v_cat]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Badge Label
            label = f"{v_cat.upper()} {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                        
            detections_list.append({
                "class": v_cat,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })
            
    total_vehicles = sum(vehicle_counts.values())
    
    # 3. Density Classification
    density_level, density_color, density_desc = calculate_traffic_density(total_vehicles)
    
    # 4. Congestion Alert
    is_congested = (total_vehicles >= 6) or (density_level == "HIGH")
    
    # 5. Traffic Signal Status
    signal_status, signal_bgr, signal_counts = detect_signal_state_hsv(img_bgr)
    
    # 6. Waiting-Time Estimation
    waiting_report = compute_waiting_time(vehicle_counts, signal_state=signal_status if signal_status != "UNKNOWN" else "RED")
    
    # Latency calculation
    latency_ms = round((time.time() - t_start) * 1000, 1)
    
    # 7. Render Telemetry HUD
    # --- Top Banner ---
    if is_congested:
        cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 220), -1)
        cv2.putText(annotated, "WARNING: JUNCTION TRAFFIC CONGESTION ALERT ACTIVE!",
                    (15, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
    else:
        cv2.rectangle(annotated, (0, 0), (w, 42), (20, 20, 20), -1)
        cv2.putText(annotated, f"VisionPort Live Portal | Density: {density_level} | Latency: {latency_ms}ms",
                    (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    
    # --- Metrics Side HUD Panel (Top-Right) ---
    hud_w, hud_h = 280, 160
    hx = w - hud_w - 15
    hy = 55
    cv2.rectangle(annotated, (hx, hy), (hx + hud_w, hy + hud_h), (25, 25, 25), -1)
    cv2.rectangle(annotated, (hx, hy), (hx + hud_w, hy + hud_h), density_color, 2)
    
    cv2.putText(annotated, "JUNCTION TELEMETRY", (hx + 12, hy + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(annotated, f"Vehicles: {total_vehicles} | {density_level}", (hx + 12, hy + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, density_color, 2)
                
    breakdown_text = f"C:{vehicle_counts['car']} B:{vehicle_counts['motorcycle']+vehicle_counts['bicycle']} Bus:{vehicle_counts['bus']} T:{vehicle_counts['truck']}"
    cv2.putText(annotated, breakdown_text, (hx + 12, hy + 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                
    # Signal Light Dot
    sig_text = f"Signal: {signal_status}"
    cv2.putText(annotated, sig_text, (hx + 38, hy + 106),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, signal_bgr, 2)
    cv2.circle(annotated, (hx + 22, hy + 102), 8, signal_bgr, -1)
    
    cv2.putText(annotated, f"Est. Wait Time: {waiting_report['total_wait_seconds']}s", (hx + 12, hy + 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 100), 2)
                
    telemetry = {
        "total_vehicles": total_vehicles,
        "breakdown": vehicle_counts,
        "density": density_level,
        "congestion_alert": is_congested,
        "signal_status": signal_status,
        "estimated_wait_time_sec": waiting_report["total_wait_seconds"],
        "latency_ms": latency_ms,
        "detections_count": len(detections_list)
    }
    
    return annotated, telemetry

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the VisionPort Live Portal HTML Dashboard."""
    html_file = "templates/index.html"
    if os.path.exists(html_file):
        with open(html_file, "r") as f:
            return f.read()
    elif os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return f.read()
    return "<h1>VisionPort Live Portal API is running. Visit /docs for documentation.</h1>"

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Standard VisionPort Detection Endpoint (Backward Compatible).
    Streams the deep-learning annotated image back to the client.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})
        
    annotated_img, telemetry = analyze_traffic_frame(img)
    
    _, encoded_img = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    headers = {
        "X-Total-Vehicles": str(telemetry["total_vehicles"]),
        "X-Traffic-Density": telemetry["density"],
        "X-Congestion-Alert": str(telemetry["congestion_alert"]),
        "X-Signal-Status": telemetry["signal_status"],
        "X-Estimated-Wait-Time": str(telemetry["estimated_wait_time_sec"]),
        "X-Inference-Latency": f"{telemetry['latency_ms']}ms",
        "Access-Control-Expose-Headers": "X-Total-Vehicles, X-Traffic-Density, X-Congestion-Alert, X-Signal-Status, X-Estimated-Wait-Time, X-Inference-Latency"
    }
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg", headers=headers)

@app.post("/analyze/")
async def analyze_traffic(file: UploadFile = File(...)):
    """
    Structured Telemetry Endpoint:
    Returns full JSON report (counts, density, signal, wait time) and base64 preview.
    """
    import base64
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})
        
    annotated_img, telemetry = analyze_traffic_frame(img)
    _, encoded_img = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_str = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
    
    return {
        "status": "success",
        "telemetry": telemetry,
        "image_base64": f"data:image/jpeg;base64,{b64_str}"
    }

@app.get("/sample/{sample_name}")
async def get_sample_inference(sample_name: str):
    """Run inference directly on a server-side sample image."""
    sample_map = {
        "sample1": "images/traffic_street.jpg",
        "sample2": "images/traffic_congested.jpg",
        "sample3": "images/traffic_signal_red.jpg",
        "sample4": "images/traffic_sample_1.jpg"
    }
    path = sample_map.get(sample_name.lower())
    if not path or not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Sample image not found"})
        
    img = cv2.imread(path)
    annotated_img, telemetry = analyze_traffic_frame(img)
    _, encoded_img = cv2.imencode(".jpg", annotated_img)
    
    headers = {
        "X-Total-Vehicles": str(telemetry["total_vehicles"]),
        "X-Traffic-Density": telemetry["density"],
        "X-Congestion-Alert": str(telemetry["congestion_alert"]),
        "X-Signal-Status": telemetry["signal_status"],
        "X-Estimated-Wait-Time": str(telemetry["estimated_wait_time_sec"]),
        "Access-Control-Expose-Headers": "*"
    }
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg", headers=headers)

@app.get("/status/")
async def system_status():
    """System diagnostic health check."""
    import torch
    device = "Apple Silicon MPS" if torch.backends.mps.is_available() else ("CUDA" if torch.cuda.is_available() else "CPU")
    return {
        "system_status": "Active",
        "model_architecture": "YOLOv8 Nano (Ultralytics)",
        "hardware_accelerator": device,
        "supported_classes": list(VEHICLE_COLORS.keys()),
        "endpoints": ["/detect/", "/analyze/", "/sample/{sample_name}", "/status/"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

