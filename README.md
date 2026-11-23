# VisionPort: Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%2FCUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Tasks](https://img.shields.io/badge/Curriculum_Tasks-20%2F20_Passed-brightgreen?style=for-the-badge)

</div>

---

### 🎓 Academic & Student Information
* **Student Name:** Kalva Amrutha
* **Student ID / Register No.:** 2520090070
* **Course:** Deep Learning (DLAC)
* **Section:** Sec-12
* **Project Title:** VisionPort – Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
* **Repository:** [https://github.com/kalvaamrutha/2520090070_DLAC-Sec-12-](https://github.com/kalvaamrutha/2520090070_DLAC-Sec-12-)

---

## 📖 Project Overview

**VisionPort** is a comprehensive, production-grade deep learning traffic analysis and junction telemetry suite. Leveraging **YOLOv8**, **OpenCV**, and a **FastAPI** backend with a modern interactive web portal, VisionPort processes video feeds and high-resolution street images in real time to provide:
1. **Multi-Class Vehicle Detection & Classification**: Cars, Motorcycles/Bikes, Buses, and Trucks.
2. **Persistent Multi-Object Tracking**: Vehicle trajectory tracking using ByteTrack with unique persistent tracking IDs.
3. **Real-Time Speed Estimation**: Calibrated speed calculation in km/h based on frame-by-frame centroid displacement.
4. **Traffic Density Classification**: Instant categorization into Low, Medium, and High traffic zones.
5. **Junction Congestion Early-Warning System**: Automated congestion alerts when junctions exceed capacity.
6. **Traffic Signal Status Recognition**: HSV color space segmentation detecting Active RED, YELLOW, or GREEN lights.
7. **Automated Queue Clearance & Waiting-Time Estimation**: Dynamic waiting-time calculation using vehicular clearance weights and signal phase delays.

---

## 🏛️ System Architecture

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
   • Cars, Buses,       • Low (0-3 vehicles)  • HSV Color Masking   • Vehicle Clearance
     Bikes, Trucks      • Medium (4-7)        • RED / YELLOW /        Weights & Signal
   • Tracking & Speed   • High (8+ vehicles)    GREEN Detection       Base Delay
```

---

## ⚡ Quick Start Guide

### 1. Environment Setup & Installation
Clone the repository and install all required dependencies:
```bash
git clone https://github.com/kalvaamrutha/2520090070_DLAC-Sec-12-.git
cd 2520090070_DLAC-Sec-12-/Skill/Skill_1_visionPort
pip install -r requirements.txt
```

### 2. Run All 20 Tasks Verification Pipeline
Execute the automated test pipeline to run and verify all 20 student tasks in sequence:
```bash
python run_all_tasks.py
```
*Expected Output:* An executive summary table displaying `✓ PASS` for all 20 tasks in ~33 seconds.

### 3. Launch the VisionPort Live Web Dashboard
Start the FastAPI server:
```bash
python main.py
```
Open your browser and navigate to:
```
http://localhost:8000
```
* **Upload Custom Images**: Drag and drop any traffic snapshot or camera capture.
* **Quick Test Buttons**: One-click test scenarios for Free-Flow Street, High Congestion, Traffic Light, and Urban Bus.
* **Live Telemetry HUD**: View total vehicle count, category breakdown, density level badge, signal status dot, and estimated waiting time.

---

## 📋 Comprehensive 20 Student Tasks Breakdown

Every task from the curriculum is implemented as an independent, modular script directly accessible in the project root and `scripts/`, with verified artifacts saved to `outputs/`:

| Task No. | Task Description | Script Name | Expected Output | Status |
|:---:|:---|:---|:---|:---:|
| **Task 1** | Install Python, OpenCV and required libraries | `task1_setup_and_verify.py` | Working Python environment report | ✅ Verified |
| **Task 2** | Capture an image of a traffic signal using camera | `task2_capture_traffic_signal.py` | Traffic signal image capture | ✅ Verified |
| **Task 3** | Read and display traffic images using OpenCV | `task3_read_display_traffic_image.py` | Image display window & metadata | ✅ Verified |
| **Task 4** | Load and play a traffic video using OpenCV | `task4_load_play_traffic_video.py` | Video playback & FPS analysis | ✅ Verified |
| **Task 5** | Extract individual frames from traffic videos | `task5_extract_video_frames.py` | Frames saved as individual images | ✅ Verified |
| **Task 6** | Detect cars in a traffic video using YOLO | `task6_detect_cars.py` | Car detection with bounding boxes | ✅ Verified |
| **Task 7** | Detect bikes, buses and trucks | `task7_detect_multiple_vehicles.py` | Multiple vehicle detection | ✅ Verified |
| **Task 8** | Display vehicle names and confidence scores | `task8_display_labels_confidence.py` | Labelled detection output | ✅ Verified |
| **Task 9** | Count the number of vehicles in a video frame | `task9_count_vehicles.py` | Vehicle count HUD panel | ✅ Verified |
| **Task 10** | Calculate traffic density based on vehicle count | `task10_traffic_density.py` | Low, medium and high traffic badge | ✅ Verified |
| **Task 11** | Detect traffic congestion at signal junctions | `task11_traffic_congestion.py` | Congestion warning alerts & log | ✅ Verified |
| **Task 12** | Track individual vehicles across video frames | `task12_track_vehicles.py` | Persistent vehicle tracking IDs | ✅ Verified |
| **Task 13** | Calculate approximate vehicle speed | `task13_calculate_vehicle_speed.py` | Speed estimation (km/h) | ✅ Verified |
| **Task 14** | Identify traffic signal status (red, yellow, green) | `task14_traffic_signal_status.py` | Signal status detection (HSV) | ✅ Verified |
| **Task 15** | Estimate traffic waiting time | `task15_traffic_waiting_time.py` | Waiting-time report (TXT & JSON) | ✅ Verified |
| **Task 16** | Count vehicles entering and leaving a traffic junction | `task16_entry_exit_vehicle_count.py` | Entry and exit vehicle count | ✅ Verified |
| **Task 17** | Detect vehicles making wrong turns at intersections | `task17_wrong_turn_detection.py` | Wrong-turn detection | ✅ Verified |
| **Task 18** | Identify vehicles moving in the wrong direction | `task18_wrong_way_vehicle_alerts.py` | Wrong-way vehicle alerts | ✅ Verified |
| **Task 19** | Detect vehicles parked in restricted traffic areas | `task19_illegal_parking_detection.py` | Illegal parking detection | ✅ Verified |
| **Task 20** | Detect two-wheelers carrying more than two persons | `task20_triple_riding_detection.py` | Triple-riding detection | ✅ Verified |

---

### Detailed Task Walkthrough & Technical Details

#### 🔹 Task 1: Environment Verification
* **Script:** `python scripts/task1_environment.py`
* **Implementation:** Verifies Python runtime, OpenCV (5.0.0), PyTorch (2.15.0 MPS/CUDA), Ultralytics YOLO (8.4.211), FastAPI, Uvicorn, Pillow, and NumPy.
* **Output Artifact:** `outputs/task1_environment_report.txt`

#### 🔹 Task 2: Camera Capture
* **Script:** `python scripts/task2_traffic_camera_capture.py`
* **Implementation:** Accesses camera device index 0 via `cv2.VideoCapture()`. Includes an automated graceful fallback to simulated traffic camera streams in headless environments.
* **Output Artifact:** `outputs/task2_traffic_signal_capture.jpg` and `images/traffic_camera_capture.jpg`

#### 🔹 Task 3: Read & Display Traffic Images
* **Script:** `python scripts/task3_read_display_traffic_image.py`
* **Implementation:** Decodes BGR images using OpenCV, extracts resolution, color channels, bit-depth, and file size, overlaying an informational telemetry header.
* **Output Artifact:** `outputs/task3_display_output.jpg`

#### 🔹 Task 4: Load & Play Traffic Video
* **Script:** `python scripts/task4_load_play_traffic_video.py`
* **Implementation:** Analyzes video properties: Frame Rate (FPS), total frame count, aspect ratio, and playback duration. Processes frame-by-frame streams with real-time timestamps.
* **Output Artifact:** `outputs/task4_video_playback_preview.jpg`

#### 🔹 Task 5: Video Frame Extraction
* **Script:** `python scripts/task5_extract_video_frames.py`
* **Implementation:** Periodically samples video feeds at specified intervals and saves timestamped high-resolution JPEG frames into the `frames/` directory.
* **Output Artifact:** `frames/frame_0001.jpg` through `frames/frame_0010.jpg`

#### 🔹 Task 6: YOLOv8 Car Detection
* **Script:** `python scripts/task6_detect_cars_yolo.py`
* **Implementation:** Runs YOLOv8 nano inference filtering specifically for COCO class ID `2` (`car`). Draws bounding boxes with confidence scores.
* **Output Artifact:** `outputs/task6_cars_detected.jpg`

#### 🔹 Task 7: Multi-Vehicle Detection
* **Script:** `python scripts/task7_detect_multiple_vehicles.py`
* **Implementation:** Detects and classifies multiple vehicle categories using color-coded bounding boxes:
  * **Cars**: Cyan / Blue `(255, 180, 0)`
  * **Motorcycles / Bikes**: Green `(0, 220, 0)`
  * **Bicycles**: Light Green `(50, 255, 150)`
  * **Buses**: Orange `(0, 140, 255)`
  * **Trucks**: Magenta `(255, 0, 200)`
* **Output Artifact:** `outputs/task7_multiple_vehicles.jpg`

#### 🔹 Task 8: Class Names & Confidence Scores
* **Script:** `python scripts/task8_display_labels_confidence.py`
* **Implementation:** Annotates each detection with contrasting dark pill badges showing the vehicle name and formatted confidence percentage (e.g., `BUS: 87.3% [0.87]`).
* **Output Artifact:** `outputs/task8_labelled_detections.jpg`

#### 🔹 Task 9: Real-Time Vehicle Counting
* **Script:** `python scripts/task9_count_vehicles.py`
* **Implementation:** Calculates total vehicle volume per frame along with individual tallies for cars, bikes, buses, and trucks, displaying an on-screen Vehicle Counter HUD.
* **Output Artifact:** `outputs/task9_vehicle_count.jpg`

#### 🔹 Task 10: Traffic Density Calculation
* **Script:** `python scripts/task10_calculate_traffic_density.py`
* **Mathematical Classification Model:**
  $$\text{Density} = \begin{cases} \text{LOW (Green)}, & \text{count} \le 3 \\ \text{MEDIUM (Amber)}, & 4 \le \text{count} \le 7 \\ \text{HIGH (Red)}, & \text{count} \ge 8 \end{cases}$$
* **Output Artifact:** `outputs/task10_traffic_density.jpg` and `outputs/task10_density_report.txt`

#### 🔹 Task 11: Junction Congestion Alert System
* **Script:** `python scripts/task11_detect_congestion_alerts.py`
* **Implementation:** Evaluates vehicle accumulation against junction capacity thresholds. Triggers high-visibility warning banners (`!!! CONGESTION ALERT !!!`) and logs timestamped events.
* **Output Artifact:** `outputs/task11_congestion_alert.jpg` and `outputs/task11_congestion_log.txt`

#### 🔹 Task 12: Multi-Object Vehicle Tracking
* **Script:** `python scripts/task12_track_vehicles.py`
* **Implementation:** Employs YOLO tracking with ByteTrack to assign persistent IDs (`ID #1`, `ID #2`) across frames, drawing centroid motion trajectory trails.
* **Output Artifact:** `outputs/task12_vehicle_tracking.jpg`

#### 🔹 Task 13: Approximate Vehicle Speed Estimation
* **Script:** `python scripts/task13_estimate_vehicle_speed.py`
* **Speed Calculation Formula:**
  $$v = \left( \frac{\Delta d_{\text{pixels}}}{\text{PPM}} \right) \cdot \frac{1}{\Delta t} \cdot 3.6 \quad [\text{km/h}]$$
  * Where $\text{PPM} = 10.0\text{ px/m}$ (Pixels Per Meter calibration) and $\Delta t = \frac{\Delta \text{frames}}{\text{FPS}}$.
* **Output Artifact:** `outputs/task13_speed_estimation.jpg`

#### 🔹 Task 14: Traffic Signal Identification
* **Script:** `python scripts/task14_identify_traffic_signal.py`
* **Implementation:** Isolates signal ROI and performs color masking in the HSV color space:
  * **RED**: $H \in [0, 10] \cup [160, 180], S \ge 120, V \ge 120$
  * **YELLOW**: $H \in [18, 35], S \ge 140, V \ge 140$
  * **GREEN**: $H \in [45, 85], S \ge 110, V \ge 110$
* **Output Artifact:** `outputs/task14_signal_status.jpg`

#### 🔹 Task 15: Queue Clearance & Waiting-Time Estimation
* **Script:** `python scripts/task15_estimate_waiting_time.py`
* **Mathematical Waiting-Time Model:**
  $$T_{\text{wait}} = T_{\text{signal\_base}} + \left( \sum_{i} N_i \cdot w_i \right) \times \mu_{\text{density}}$$
  * Clearance weights $w_i$: Car = 2.5s, Bike = 1.5s, Bus = 5.0s, Truck = 6.0s.
  * Base signal delay $T_{\text{signal\_base}}$: RED = 35s, YELLOW = 10s, GREEN = 0s.
  * Density multiplier $\mu_{\text{density}}$: LOW = 1.0x, MEDIUM = 1.2x, HIGH = 1.5x.
* **Output Artifact:** `outputs/task15_waiting_time_report.txt` and `outputs/task15_waiting_time_report.json`

#### 🔹 Task 16: Entry and Exit Vehicle Count
* **Script:** `python task16_entry_exit_vehicle_count.py`
* **Mathematical Tripwire Formulation:**
  A vehicle with centroid trajectory $P_t = (x_t, y_t)$ crosses horizontal counting tripwires $Y_{\text{entry}}$ and $Y_{\text{exit}}$:
  $$\text{Entry Trigger} \iff y_{t-1} < Y_{\text{entry}} \le y_t, \quad \text{Exit Trigger} \iff y_{t-1} < Y_{\text{exit}} \le y_t$$
  Each crossing vehicle is tracked with a persistent ID to guarantee single-event counting:
  $$N_{\text{total}} = N_{\text{entry}} + N_{\text{exit}}$$
* **Output Artifact:** `outputs/task16_entry_exit_count.jpg`

#### 🔹 Task 17: Wrong-Turn Detection
* **Script:** `python task17_wrong_turn_detection.py`
* **Trajectory Heading Analysis:**
  Computes the instantaneous velocity vector and turn angle across a window of $k$ consecutive frames:
  $$\vec{v} = (\Delta x, \Delta y) = (x_t - x_{t-k}, y_t - y_{t-k}), \quad \theta_{\text{turn}} = \arctan2(\Delta y, \Delta x) \times \frac{180}{\pi}$$
  If a vehicle in a designated straight-only lane initiates an unauthorized lateral turn $(|\Delta x| > \tau_{\text{lateral}})$ into cross-traffic, a high-severity alert is dispatched.
* **Output Artifact:** `outputs/task17_wrong_turn_detection.jpg`

#### 🔹 Task 18: Wrong-Way Vehicle Alerts
* **Script:** `python task18_wrong_way_vehicle_alerts.py`
* **Directional Alignment Model:**
  Given the legal lane flow unit vector $\vec{u}_{\text{flow}} = (0, 1)$ (downward perspective), the vehicle's normalized motion vector $\hat{v} = \frac{\vec{v}}{\|\vec{v}\|}$ is projected via dot product:
  $$\cos(\alpha) = \hat{v} \cdot \vec{u}_{\text{flow}}$$
  A vehicle is classified as heading in the wrong direction when $\cos(\alpha) < 0$ ($\Delta y < 0$ in forward lanes), activating an audible/visual alarm overlay.
* **Output Artifact:** `outputs/task18_wrong_way_alerts.jpg`

#### 🔹 Task 19: Illegal Parking Detection
* **Script:** `python task19_illegal_parking_detection.py`
* **Spatio-Temporal Stationary Constraint:**
  For a polygon-defined restricted zone $Z_{\text{restricted}}$, a vehicle with centroid $c(t)$ and bounding box $B(t)$ is evaluated:
  $$c(t) \in Z_{\text{restricted}} \quad \text{and} \quad \|\vec{v}(t)\| < \epsilon \quad \forall t \in [t_{\text{stop}}, t_{\text{stop}} + T_{\text{threshold}}]$$
  If $T_{\text{stop}} \ge 2.0\text{ seconds}$, an illegal parking violation citation is logged with bounding box highlighting.
* **Output Artifact:** `outputs/task19_illegal_parking.jpg`

#### 🔹 Task 20: Triple-Riding Detection
* **Script:** `python task20_triple_riding_detection.py`
* **Rider-to-Vehicle Spatial Association Model:**
  For each detected two-wheeler (motorcycle/bicycle) bounding box $B_{\text{bike}} = [x_1, y_1, x_2, y_2]$, an expanded vertical rider zone $Z_{\text{rider}}$ is established:
  $$Z_{\text{rider}} = [x_1 - \delta_x, y_1 - 0.8(y_2 - y_1), x_2 + \delta_x, y_2]$$
  The total number of riders $K_{\text{riders}}$ is computed by summing all detected person centroids $c_p$ inside $Z_{\text{rider}}$:
  $$K_{\text{riders}} = \sum_{p \in \text{Persons}} \mathbb{I}(c_p \in Z_{\text{rider}})$$
  If $K_{\text{riders}} > 2$, a **Triple-Riding Safety Violation** is flagged.
* **Output Artifact:** `outputs/task20_triple_riding.jpg`

---

## 📡 REST API Reference

### 1. Stream Detection (`POST /detect/`)
Uploads an image and streams back real-time annotated JPEG bytes with telemetry response headers.
```bash
curl -X POST "http://localhost:8000/detect/" \
     -H "accept: image/jpeg" \
     -F "file=@images/traffic_street.jpg" \
     --output response_annotated.jpg
```
**Response Headers:**
* `X-Total-Vehicles`: `3`
* `X-Traffic-Density`: `LOW`
* `X-Congestion-Alert`: `False`
* `X-Signal-Status`: `RED`
* `X-Estimated-Wait-Time`: `44.7`

### 2. Structured Telemetry Analysis (`POST /analyze/`)
Uploads an image and receives a full JSON telemetry payload with base64 preview:
```bash
curl -X POST "http://localhost:8000/analyze/" \
     -F "file=@images/traffic_street.jpg"
```
**JSON Response:**
```json
{
  "status": "success",
  "telemetry": {
    "total_vehicles": 3,
    "breakdown": {
      "car": 1,
      "motorcycle": 0,
      "bicycle": 1,
      "bus": 0,
      "truck": 1
    },
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

### 3. Server Sample Evaluation (`GET /sample/{sample_name}`)
Test pre-loaded scenarios without uploading files:
* `/sample/sample1`: Free-Flow Traffic Street
* `/sample/sample2`: Heavy Junction Congestion
* `/sample/sample3`: Traffic Signal (Red Light)
* `/sample/sample4`: Urban Bus & Pedestrians

### 4. Health & Hardware Check (`GET /status/`)
```json
{
  "system_status": "Active",
  "model_architecture": "YOLOv8 Nano (Ultralytics)",
  "hardware_accelerator": "Apple Silicon MPS",
  "supported_classes": ["car", "motorcycle", "bicycle", "bus", "truck"],
  "endpoints": ["/detect/", "/analyze/", "/sample/{sample_name}", "/status/"]
}
```

---

## 📁 Repository Directory Structure

```
Skill/Skill_1_visionPort/
├── main.py                             # Enhanced FastAPI backend server
├── index.html / templates/index.html   # VisionPort Live Web Dashboard
├── run_all_tasks.py                    # Master pipeline runner for Tasks 1 to 15
├── requirements.txt                    # Python package dependencies
├── README.md                           # Documentation & report
├── .gitignore                          # Git ignore specification
├── scripts/                            # 15 Independent Task Modules
│   ├── task1_environment.py            # Task 1: Environment verification
│   ├── task2_traffic_camera_capture.py # Task 2: Camera capture & simulation
│   ├── task3_read_display_traffic_image.py # Task 3: Image reading & display
│   ├── task4_load_play_traffic_video.py# Task 4: Video loading & playback
│   ├── task5_extract_video_frames.py   # Task 5: Frame extraction
│   ├── task6_detect_cars_yolo.py       # Task 6: YOLO car detection
│   ├── task7_detect_multiple_vehicles.py # Task 7: Multi-vehicle detection
│   ├── task8_display_labels_confidence.py# Task 8: Class labels & confidence
│   ├── task9_count_vehicles.py         # Task 9: Vehicle counting HUD
│   ├── task10_calculate_traffic_density.py # Task 10: Traffic density calculation
│   ├── task11_detect_congestion_alerts.py # Task 11: Junction congestion alerts
│   ├── task12_track_vehicles.py        # Task 12: ByteTrack multi-object tracking
│   ├── task13_estimate_vehicle_speed.py# Task 13: Vehicle speed estimation
│   ├── task14_identify_traffic_signal.py # Task 14: Traffic signal status (HSV)
│   ├── task15_estimate_waiting_time.py # Task 15: Waiting time estimation report
│   └── generate_sample_assets.py       # Sample image/video generator
├── models/                             # Pre-trained deep learning weights
│   └── yolov8n.pt                      # YOLOv8 Nano weights
├── images/                             # Test images & traffic signal captures
├── videos/                             # Sample traffic videos (MP4)
├── frames/                             # Extracted video frames from Task 5
├── outputs/                            # Verified task outputs and logs
└── static/                             # Web assets
```

---

## 💻 Tech Stack & Dependencies

* **Language**: Python 3.14+
* **Deep Learning Framework**: PyTorch with Apple Silicon MPS / CUDA GPU acceleration
* **Object Detection & Tracking**: Ultralytics YOLOv8 Nano (`yolov8n.pt`) with ByteTrack
* **Computer Vision**: OpenCV (`cv2`) 5.0.0
* **API Backend**: FastAPI & Starlette
* **ASGI Server**: Uvicorn
* **Data & Image Processing**: NumPy, Pillow (`PIL`), Matplotlib

---

## 👩‍💻 Author & Submission Details

* **Student Name:** Kalva Amrutha
* **Register Number / ID:** 2520090070
* **Course:** Deep Learning (DLAC) — Section 12
* **Submission Date:** November 2026
* **GitHub Repository:** [https://github.com/kalvaamrutha/2520090070_DLAC-Sec-12-](https://github.com/kalvaamrutha/2520090070_DLAC-Sec-12-)
