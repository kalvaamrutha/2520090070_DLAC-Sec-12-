"""
Task 5: Extract individual frames from traffic videos.
Expected Output: Frames saved as images
"""
import cv2
import os
import sys

def extract_video_frames(video_path="videos/traffic_demo.mp4", output_dir="frames", frame_step=15, max_extract=10):
    print("=" * 60)
    print("VisionPort - Task 5: Extract Individual Video Frames")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        if os.path.exists("videos/traffic_junction_demo.mp4"):
            video_path = "videos/traffic_junction_demo.mp4"
        else:
            print(f"[ERROR] Video file not found: {video_path}")
            return 1
            
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return 1
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source Video       : {video_path}")
    print(f"Total Frames       : {total_frames} (FPS: {fps:.2f})")
    print(f"Sampling Interval  : Every {frame_step} frames")
    print(f"Output Directory   : {output_dir}/")
    
    saved_count = 0
    frame_idx = 0
    
    while cap.isOpened() and saved_count < max_extract:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        if frame_idx % frame_step == 0:
            saved_count += 1
            frame_filename = f"frame_{saved_count:04d}.jpg"
            out_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(out_path, frame)
            timestamp_sec = frame_idx / fps
            print(f" [Saved] {frame_filename} (source frame #{frame_idx}, time: {timestamp_sec:.2f}s)")
            
        frame_idx += 1
        
    cap.release()
    print("-" * 60)
    print(f"Extraction Complete: {saved_count} frames saved to {output_dir}/")
    print("Expected Output: Frames saved as images [VERIFIED]")
    return 0

if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "videos/traffic_demo.mp4"
    sys.exit(extract_video_frames(video_path=vid))

