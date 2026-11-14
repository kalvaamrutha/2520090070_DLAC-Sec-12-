"""
VisionPort Master Runner
Executes all 15 student tasks sequentially and validates their expected outputs in outputs/.
"""
import sys
import subprocess
import time
import os

TASKS = [
    (1,  "Environment Verification",       "scripts/task1_setup_and_verify.py",         "outputs/task1_opencv_test.jpg"),
    (2,  "Traffic Image Capture",          "scripts/task2_capture_traffic_signal.py",   "outputs/captured_traffic_signal.jpg"),
    (3,  "Read & Display Traffic Image",   "scripts/task3_read_display_traffic_image.py","outputs/task3_display_output.jpg"),
    (4,  "Load & Play Traffic Video",      "scripts/task4_load_play_traffic_video.py",  "outputs/task4_playback_preview.jpg"),
    (5,  "Extract Video Frames",           "scripts/task5_extract_video_frames.py",     "extracted_frames/frame_0001.jpg"),
    (6,  "Detect Cars with YOLO",          "scripts/task6_detect_cars.py",              "outputs/task6_cars_detected.jpg"),
    (7,  "Detect Multiple Vehicles",       "scripts/task7_detect_multiple_vehicles.py", "outputs/task7_multiple_vehicles.jpg"),
    (8,  "Display Labels & Confidence",    "scripts/task8_display_labels_confidence.py","outputs/task8_labelled_detections.jpg"),
    (9,  "Vehicle Counting",               "scripts/task9_count_vehicles.py",           "outputs/task9_vehicle_count.jpg"),
    (10, "Traffic Density Calculation",    "scripts/task10_traffic_density.py",         "outputs/task10_traffic_density.jpg"),
    (11, "Junction Congestion Alerts",     "scripts/task11_traffic_congestion.py",      "outputs/task11_congestion_alert.jpg"),
    (12, "Vehicle Tracking IDs",           "scripts/task12_track_vehicles.py",          "outputs/task12_vehicle_tracking.jpg"),
    (13, "Speed Estimation",               "scripts/task13_calculate_vehicle_speed.py", "outputs/task13_speed_estimation.jpg"),
    (14, "Traffic Signal Identification",  "scripts/task14_traffic_signal_status.py",   "outputs/task14_signal_status.jpg"),
    (15, "Waiting-Time Estimation Report", "scripts/task15_traffic_waiting_time.py",    "outputs/task15_waiting_time_report.txt"),
]

def run_pipeline():
    print("=" * 80)
    print("       VISIONPORT - ALL 15 TASKS EXECUTION & VERIFICATION PIPELINE       ")
    print("=" * 80)
    
    python_bin = sys.executable
    results = []
    start_all = time.time()
    
    for num, name, script, expected_file in TASKS:
        t0 = time.time()
        print(f"\n[RUNNING] Task {num:>2}: {name} ({script})...")
        cmd = [python_bin, script]
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        
        file_ok = os.path.exists(expected_file)
        success = (proc.returncode == 0) and file_ok
        
        status_label = "PASS" if success else "FAIL"
        results.append((num, name, status_label, f"{elapsed:.2f}s", expected_file))
        
        if success:
            print(f" -> Result: PASS ({elapsed:.2f}s) | Verified: {expected_file}")
        else:
            print(f" -> Result: FAIL ({elapsed:.2f}s)")
            if proc.returncode != 0:
                print(f"    Exit code: {proc.returncode}")
                print(f"    Stderr: {proc.stderr[-200:]}")
            if not file_ok:
                print(f"    Missing expected output file: {expected_file}")
                
    total_elapsed = time.time() - start_all
    
    # Print Executive Summary Table
    print("\n" + "=" * 80)
    print("                       VISIONPORT EXECUTION SUMMARY TABLE                       ")
    print("=" * 80)
    print(f"{'Task':<6} | {'Task Description':<32} | {'Status':<6} | {'Time':<7} | {'Verified Output'}")
    print("-" * 80)
    
    passed_count = sum(1 for r in results if r[2] == "PASS")
    for num, name, st, el, out_f in results:
        mark = "✓" if st == "PASS" else "✗"
        print(f"T{num:<5} | {name:<32} | {mark} {st:<4} | {el:<7} | {out_f}")
        
    print("-" * 80)
    print(f"Total Completed: {passed_count}/{len(TASKS)} Tasks Passed in {total_elapsed:.2f}s")
    print("=" * 80)
    return 0 if passed_count == len(TASKS) else 1

if __name__ == "__main__":
    sys.exit(run_pipeline())
