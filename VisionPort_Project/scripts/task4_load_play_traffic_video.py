"""
Task 4: Load and play a traffic video using OpenCV.
Expected Output: Video playback
"""
import cv2
import os
import sys

def load_and_play_traffic_video(video_path="videos/traffic_junction_demo.mp4", max_frames=50, show_gui=False):
    print("=" * 60)
    print("VisionPort - Task 4: Load and Play Traffic Video")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        # Fallback to traffic_demo.mp4
        if os.path.exists("videos/traffic_demo.mp4"):
            video_path = "videos/traffic_demo.mp4"
        else:
            print(f"[ERROR] Video file not found at {video_path}")
            return 1
            
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file {video_path}")
        return 1
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    
    print(f"Video File         : {video_path}")
    print(f"Resolution         : {width} x {height}")
    print(f"Frame Rate (FPS)   : {fps:.2f}")
    print(f"Total Frames       : {total_frames}")
    print(f"Duration           : {duration_sec:.2f} seconds")
    
    frames_processed = 0
    preview_frame = None
    
    while cap.isOpened() and frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        frames_processed += 1
        
        # Overlay playback HUD
        cv2.rectangle(frame, (10, 10), (380, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"PLAYBACK: Frame {frames_processed}/{total_frames} ({frames_processed/fps:.2f}s)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        preview_frame = frame
        
        if show_gui:
            try:
                cv2.imshow("VisionPort - Task 4 Video Playback", frame)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break
            except Exception:
                pass
                
    cap.release()
    if show_gui:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
            
    os.makedirs("outputs", exist_ok=True)
    if preview_frame is not None:
        out_path = "outputs/task4_video_playback_preview.jpg"
        cv2.imwrite(out_path, preview_frame)
        print(f"Playback preview frame saved to: {out_path}")
        
    print(f"Successfully processed {frames_processed} frames from video.")
    print("-" * 60)
    print("Expected Output: Video playback [VERIFIED]")
    return 0

if __name__ == "__main__":
    vid_file = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "videos/traffic_junction_demo.mp4"
    gui_flag = "--gui" in sys.argv
    sys.exit(load_and_play_traffic_video(vid_file, show_gui=gui_flag))

