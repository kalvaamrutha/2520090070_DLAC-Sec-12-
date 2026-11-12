"""
Script to prepare and generate sample traffic images, signals, and demo videos
for VisionPort testing and verification across all 15 tasks.
"""
import os
import cv2
import numpy as np

os.makedirs("images", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("frames", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def draw_traffic_light(img, center_x, center_y, state="red", scale=1.0):
    """Draw a realistic traffic signal on an image."""
    w = int(40 * scale)
    h = int(100 * scale)
    r = int(12 * scale)
    
    # Housing
    top_left = (center_x - w // 2, center_y - h // 2)
    bottom_right = (center_x + w // 2, center_y + h // 2)
    cv2.rectangle(img, top_left, bottom_right, (30, 30, 30), -1)
    cv2.rectangle(img, top_left, bottom_right, (180, 180, 180), 2)
    
    # Visors
    positions = [
        (center_x, center_y - int(30 * scale)),  # Red
        (center_x, center_y),                    # Yellow
        (center_x, center_y + int(30 * scale))   # Green
    ]
    
    colors = {
        "red": ((40, 40, 120), (30, 30, 255)),     # (dark BGR, bright BGR)
        "yellow": ((30, 100, 100), (30, 230, 240)),
        "green": ((40, 100, 40), (40, 240, 40))
    }
    
    states = ["red", "yellow", "green"]
    for i, s in enumerate(states):
        pos = positions[i]
        dark_col, bright_col = colors[s]
        if s == state.lower():
            # Glowing light
            cv2.circle(img, pos, r + 3, bright_col, 2)
            cv2.circle(img, pos, r, bright_col, -1)
            cv2.circle(img, pos, r // 2, (255, 255, 255), -1)
        else:
            cv2.circle(img, pos, r, dark_col, -1)
            cv2.circle(img, pos, r, (60, 60, 60), 1)

def generate_signal_images():
    """Generate sample images containing RED, YELLOW, and GREEN signals."""
    for state in ["red", "yellow", "green"]:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Background: Sky & Road
        img[:280, :] = [180, 150, 120]  # Sky (BGR)
        img[280:, :] = [70, 70, 70]     # Road
        
        # Lane markings
        cv2.line(img, (0, 380), (640, 380), (255, 255, 255), 3)
        cv2.line(img, (320, 280), (320, 480), (0, 220, 255), 2) # Yellow median
        
        # Signal Pole
        cv2.rectangle(img, (315, 120), (325, 280), (100, 100, 100), -1)
        # Signal head
        draw_traffic_light(img, 320, 110, state=state, scale=1.4)
        
        # Label
        cv2.putText(img, f"Traffic Signal: {state.upper()}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        path = f"images/traffic_signal_{state}.jpg"
        cv2.imwrite(path, img)
        print(f"Generated {path}")

def generate_congested_scene():
    """
    Create a composite realistic traffic junction image with multiple vehicles
    and a traffic light for high density / congestion testing.
    """
    # Start with an existing base image if present, or create a road scene
    base_path = "images/traffic_street.jpg"
    if os.path.exists(base_path):
        img = cv2.imread(base_path)
    else:
        img = np.full((540, 960, 3), 60, dtype=np.uint8)
    
    # Add a traffic signal at the top-right
    h, w = img.shape[:2]
    draw_traffic_light(img, w - 80, 100, state="red", scale=1.2)
    
    path = "images/traffic_congested.jpg"
    cv2.imwrite(path, img)
    print(f"Generated {path}")

def generate_traffic_junction_video():
    """
    Generate a 10-second traffic junction simulation video (20 fps)
    showing vehicles moving, stopping at a red signal, and moving on green.
    """
    video_path = "videos/traffic_junction_demo.mp4"
    width, height = 800, 500
    fps = 20
    total_frames = fps * 8  # 8 seconds = 160 frames
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Simulation vehicles: [x, y, speed, type, color, length, width]
    vehicles = [
        {"id": 1, "x": 50,  "y": 300, "vx": 4.5, "type": "car",   "color": (200, 50, 50),   "len": 60, "wid": 35},
        {"id": 2, "x": 180, "y": 300, "vx": 4.5, "type": "car",   "color": (50, 180, 50),   "len": 55, "wid": 35},
        {"id": 3, "x": 30,  "y": 370, "vx": 3.2, "type": "bus",   "color": (30, 150, 230),  "len": 110,"wid": 45},
        {"id": 4, "x": 220, "y": 420, "vx": 5.0, "type": "bike",  "color": (220, 220, 40),  "len": 35, "wid": 18},
        {"id": 5, "x": 350, "y": 370, "vx": 3.0, "type": "truck", "color": (150, 75, 0),    "len": 120,"wid": 48},
        {"id": 6, "x": 420, "y": 300, "vx": 4.0, "type": "car",   "color": (180, 50, 180),  "len": 58, "wid": 35},
    ]
    
    stop_line_x = 550
    
    for f in range(total_frames):
        # Frame canvas
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background: Sky & roadside
        frame[:220, :] = [180, 140, 100]  # Light sky
        frame[220:, :] = [60, 60, 60]     # Asphalt road
        
        # Sidewalk & grass
        frame[200:220, :] = [90, 130, 90]
        
        # Lane dividers (dashed lines)
        for x in range(0, width, 40):
            cv2.line(frame, (x, 340), (x + 20, 340), (255, 255, 255), 2)
            cv2.line(frame, (x, 410), (x + 20, 410), (255, 255, 255), 2)
        
        # Stop line
        cv2.line(frame, (stop_line_x, 220), (stop_line_x, 500), (255, 255, 255), 4)
        cv2.putText(frame, "STOP LINE", (stop_line_x - 90, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Signal state timing:
        # 0-50 frames: GREEN
        # 51-70 frames: YELLOW
        # 71-130 frames: RED
        # 131-160 frames: GREEN
        if f < 50:
            signal_state = "green"
        elif f < 70:
            signal_state = "yellow"
        elif f < 130:
            signal_state = "red"
        else:
            signal_state = "green"
        
        # Draw signal pole and signal
        cv2.rectangle(frame, (stop_line_x + 35, 120), (stop_line_x + 45, 220), (90, 90, 90), -1)
        draw_traffic_light(frame, stop_line_x + 40, 110, state=signal_state, scale=1.1)
        
        # Update and draw vehicles
        for v in vehicles:
            front_x = v["x"] + v["len"]
            
            # Check stopping logic if signal is RED/YELLOW
            should_stop = (signal_state in ["red", "yellow"]) and (front_x < stop_line_x) and (front_x > stop_line_x - 180)
            
            if should_stop:
                current_speed = max(0.0, v["vx"] * (stop_line_x - front_x) / 180.0)
            else:
                current_speed = v["vx"]
            
            v["x"] += current_speed
            if v["x"] > width:
                v["x"] = -v["len"] - np.random.randint(10, 80)
            
            # Draw vehicle body
            vx, vy = int(v["x"]), int(v["y"])
            vl, vw = int(v["len"]), int(v["wid"])
            cv2.rectangle(frame, (vx, vy), (vx + vl, vy + vw), v["color"], -1)
            cv2.rectangle(frame, (vx, vy), (vx + vl, vy + vw), (240, 240, 240), 2)
            
            # Windshield / windows
            cv2.rectangle(frame, (vx + vl - 15, vy + 4), (vx + vl - 5, vy + vw - 4), (220, 220, 220), -1)
            
            # Wheels
            cv2.rectangle(frame, (vx + 5, vy + vw), (vx + 18, vy + vw + 3), (10, 10, 10), -1)
            cv2.rectangle(frame, (vx + vl - 20, vy + vw), (vx + vl - 7, vy + vw + 3), (10, 10, 10), -1)
            
            # Label
            cv2.putText(frame, f"{v['type'].upper()} #{v['id']}", (vx, vy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Time and Status HUD
        cv2.putText(frame, f"VisionPort Traffic Junction | Frame {f+1}/{total_frames}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Signal Phase: {signal_state.upper()}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Generated {video_path} ({total_frames} frames)")

if __name__ == "__main__":
    generate_signal_images()
    generate_congested_scene()
    generate_traffic_junction_video()
    print("All sample assets prepared successfully!")

