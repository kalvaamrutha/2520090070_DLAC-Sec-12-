# VisionPort – Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning

VisionPort is an end-to-end intelligent traffic monitoring and junction telemetry system built using **FastAPI**, **YOLOv8 (Ultralytics)**, and **OpenCV**. It supports real-time multi-class vehicle detection, persistent multi-object tracking, speed estimation, traffic density classification, junction congestion alerting, traffic signal status identification, and automated queue waiting-time estimation.

---

## 📋 System Architecture

```
                                  +---------------------------------------+
                                  |     VisionPort Live Web Dashboard     |
                                  |       (HTML5 / Modern JS / CSS)       |
                                  +---------------------------------------+
                                             |                 ^
                               (Multipart Upload)       (Real-Time Stream &
                                             |           Telemetry Headers)
                                             v                 |
+------------------------------------------------------------------------------------+
|                                FastAPI Backend (main.py)                           |
+------------------------------------------------------------------------------------+
|  • /detect/   : Stream annotated JPEG image with telemetry HTTP response headers   |
|  • /analyze/  : Structured JSON payload (counts, breakdown, density, signal, wait) |
|  • /sample/   : Server-side traffic test samples (Free Flow, Congested, Signal)    |
|  • /status/   : System health & hardware accelerator diagnostics                   |
+------------------------------------------------------------------------------------+
          |                     |                      |                     |
          v                     v                      v                     v
   [YOLOv8 Detection]   [Traffic Density]     [Signal Recognition]  [Waiting Time Engine]
   • Cars, Buses,       • Low (<4 vehicles)   • HSV Color Masking   • Vehicle Clearance
     Bikes, Trucks      • Medium (4-7)        • RED / YELLOW /        Weights & Signal
   • Tracking & Speed   • High (8+ vehicles)    GREEN Detection       Base Delay
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup
Clone the repository and install required packages:
```bash
pip install -r requirements.txt
```

### 2. Run All 15 Tasks in One Command
To execute and verify all 15 curriculum tasks automatically:
```bash
python run_all_tasks.py
```
This runs each task script sequentially, verifies that the expected output files are generated, and displays a summary table.

### 3. Launch the VisionPort Live Web Portal
Start the FastAPI server:
```bash
python main.py
```
Open your browser and navigate to:
```
http://localhost:8000
```
- Upload your own traffic images or click on the **Quick Test Samples** buttons (Free-Flow, Congestion, Traffic Signal, Urban Bus).
- View live annotated bounding boxes, total vehicle counts, breakdown, density gauges, signal status, and estimated wait times.

---

## 📁 Repository Structure

```
VisionPort_python/
├── main.py                             # Enhanced FastAPI backend server
├── run_all_tasks.py                    # Master pipeline runner for all 15 tasks
├── requirements.txt                    # Project dependencies
├── index.html / templates/index.html   # VisionPort Live Portal Web Dashboard
├── scripts/                            # Modular scripts for Tasks 1 to 15
│   ├── task1_environment.py            # Task 1: Environment verification
│   ├── task2_traffic_camera_capture.py # Task 2: Camera image capture
│   ├── task3_read_display_traffic_image.py # Task 3: Read & display image
│   ├── task4_load_play_traffic_video.py# Task 4: Video loading & playback
│   ├── task5_extract_video_frames.py   # Task 5: Frame extraction
│   ├── task6_detect_cars_yolo.py       # Task 6: YOLO car detection
│   ├── task7_detect_multiple_vehicles.py # Task 7: Multi-vehicle detection
│   ├── task8_display_labels_confidence.py# Task 8: Class names & confidences
│   ├── task9_count_vehicles.py         # Task 9: Vehicle counting HUD
│   ├── task10_calculate_traffic_density.py # Task 10: Traffic density calculation
│   ├── task11_detect_congestion_alerts.py # Task 11: Junction congestion alerts
│   ├── task12_track_vehicles.py        # Task 12: ByteTrack multi-object tracking
│   ├── task13_estimate_vehicle_speed.py# Task 13: Vehicle speed estimation
│   ├── task14_identify_traffic_signal.py # Task 14: Traffic signal status (HSV)
│   ├── task15_estimate_waiting_time.py # Task 15: Waiting time estimation report
│   └── generate_sample_assets.py       # Sample asset generator
├── images/                             # Test traffic images & signal captures
├── videos/                             # Sample traffic video files
├── frames/                             # Extracted video frames from Task 5
├── models/                             # Pre-trained deep learning weights (yolov8n.pt)
└── outputs/                            # Generated artifacts & verification reports
```

---

## 🎯 Task Breakdown and Verification

| Task No. | Task Description | Script Name | Expected Output | Status |
|:---:|:---|:---|:---|:---:|
| **1** | Install Python, OpenCV and required libraries | `scripts/task1_environment.py` | Working Python environment | Verified |
| **2** | Capture an image of a traffic signal using camera | `scripts/task2_traffic_camera_capture.py` | Traffic image capture | Verified |
| **3** | Read and display traffic images using OpenCV | `scripts/task3_read_display_traffic_image.py` | Image display window | Verified |
| **4** | Load and play a traffic video using OpenCV | `scripts/task4_load_play_traffic_video.py` | Video playback | Verified |
| **5** | Extract individual frames from traffic videos | `scripts/task5_extract_video_frames.py` | Frames saved as images | Verified |
| **6** | Detect cars in a traffic video using YOLO | `scripts/task6_detect_cars_yolo.py` | Car detection with bounding boxes | Verified |
| **7** | Detect bikes, buses and trucks | `scripts/task7_detect_multiple_vehicles.py` | Multiple vehicle detection | Verified |
| **8** | Display vehicle names and confidence scores | `scripts/task8_display_labels_confidence.py` | Labelled detection output | Verified |
| **9** | Count the number of vehicles in a video frame | `scripts/task9_count_vehicles.py` | Vehicle count | Verified |
| **10** | Calculate traffic density based on vehicle count | `scripts/task10_calculate_traffic_density.py` | Low, medium and high traffic | Verified |
| **11** | Detect traffic congestion at signal junctions | `scripts/task11_detect_congestion_alerts.py` | Congestion alerts | Verified |
| **12** | Track individual vehicles across video frames | `scripts/task12_track_vehicles.py` | Vehicle tracking IDs | Verified |
| **13** | Calculate approximate vehicle speed | `scripts/task13_estimate_vehicle_speed.py` | Speed estimation | Verified |
| **14** | Identify traffic signal status (red, yellow, green) | `scripts/task14_identify_traffic_signal.py` | Signal status detection | Verified |
| **15** | Estimate traffic waiting time | `scripts/task15_estimate_waiting_time.py` | Waiting-time report | Verified |

---

## 📡 API Reference

### `POST /detect/`
Accepts an image file (`multipart/form-data`) and streams back the real-time YOLOv8 annotated image with telemetry response headers (`X-Total-Vehicles`, `X-Traffic-Density`, `X-Congestion-Alert`, `X-Signal-Status`, `X-Estimated-Wait-Time`).

### `POST /analyze/`
Accepts an image file and returns a structured JSON payload:
```json
{
  "status": "success",
  "telemetry": {
    "total_vehicles": 3,
    "breakdown": { "car": 1, "motorcycle": 0, "bicycle": 1, "bus": 0, "truck": 1 },
    "density": "LOW",
    "congestion_alert": false,
    "signal_status": "RED",
    "estimated_wait_time_sec": 44.7,
    "latency_ms": 36.4,
    "detections_count": 3
  },
  "image_base64": "data:image/jpeg;base64,..."
}
```

### `GET /sample/{sample_name}`
Quick inference on sample traffic assets (`sample1`, `sample2`, `sample3`, `sample4`).

### `GET /status/`
Health check endpoint reporting model architecture, active compute device (MPS/CUDA/CPU), and loaded endpoints.

---

## 📤 GitHub Upload Instructions

To upload your project to GitHub by the end of the session, run the following commands in your terminal:

```bash
# 1. Initialize git repository
git init

# 2. Stage all project files
git add .

# 3. Commit the changes
git commit -m "Complete VisionPort tasks 1-15 and enhanced FastAPI dashboard"

# 4. Set main branch
git branch -M main

# 5. Link your GitHub remote repository (replace with your repo URL)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/VisionPort.git

# 6. Push to GitHub
git push -u origin main
```

