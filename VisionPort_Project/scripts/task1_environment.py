"""
Task 1: Install Python, OpenCV and required libraries.
Expected Output: Working Python environment
"""
import sys
import platform
import os

def check_environment():
    print("=" * 60)
    print("VisionPort - Task 1: Environment Verification")
    print("=" * 60)
    
    print(f"Operating System   : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Executable  : {sys.executable}")
    print(f"Python Version     : {sys.version.split()[0]}")
    
    modules = {
        "cv2": "OpenCV",
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "ultralytics": "Ultralytics YOLO",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn"
    }
    
    success = True
    report_lines = [
        "VisionPort Python Environment Diagnostic Report",
        f"Generated: {sys.version}",
        "-" * 50
    ]
    
    for mod_name, label in modules.items():
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Installed")
            status_line = f"[PASS] {label:<18}: {ver}"
            print(status_line)
            report_lines.append(status_line)
        except ImportError as e:
            status_line = f"[FAIL] {label:<18}: NOT INSTALLED ({e})"
            print(status_line)
            report_lines.append(status_line)
            success = False
            
    # Check GPU / MPS acceleration
    try:
        import torch
        if torch.cuda.is_available():
            device_info = f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            device_info = "Apple Silicon MPS (Metal Performance Shaders)"
        else:
            device_info = "CPU (standard inference mode)"
        print(f"Inference Device   : {device_info}")
        report_lines.append(f"Inference Device   : {device_info}")
    except Exception:
        pass
        
    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/task1_environment_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
        
    print("-" * 60)
    if success:
        print("Expected Output: Working Python environment [VERIFIED]")
        print(f"Report saved to: {report_path}")
        return 0
    else:
        print("Expected Output: Incomplete Python environment [FAILED]")
        return 1

if __name__ == "__main__":
    sys.exit(check_environment())

