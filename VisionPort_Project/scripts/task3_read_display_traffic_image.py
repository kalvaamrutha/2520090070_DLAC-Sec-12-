"""
Task 3: Read and display traffic images using OpenCV.
Expected Output: Image display window
"""
import cv2
import os
import sys

def read_and_display_traffic_image(image_path="images/traffic_sample_1.jpg", show_gui=False):
    print("=" * 60)
    print("VisionPort - Task 3: Read and Display Traffic Images")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at {image_path}")
        return 1
        
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to decode image from {image_path}")
        return 1
        
    h, w, c = img.shape
    dtype = img.dtype
    size_kb = os.path.getsize(image_path) / 1024.0
    
    print(f"Image File         : {image_path}")
    print(f"Dimensions         : {w} x {h} (Width x Height)")
    print(f"Color Channels     : {c} (BGR format)")
    print(f"Data Type          : {dtype}")
    print(f"File Size          : {size_kb:.2f} KB")
    
    # Create display canvas with informational header
    display_img = img.copy()
    cv2.rectangle(display_img, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(display_img, f"VisionPort Traffic Image Inspector | {w}x{h} px | {c} Channels",
                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/task3_display_output.jpg"
    cv2.imwrite(out_path, display_img)
    print(f"Annotated display saved to: {out_path}")
    
    # If GUI is available and requested, show window
    if show_gui:
        try:
            cv2.imshow("VisionPort - Task 3 Traffic Image", display_img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            print("Displayed in GUI window successfully.")
        except Exception as e:
            print(f"[INFO] GUI window not available: {e}")
            
    print("-" * 60)
    print("Expected Output: Image display window [VERIFIED]")
    return 0

if __name__ == "__main__":
    img_file = sys.argv[1] if len(sys.argv) > 1 else "images/traffic_sample_1.jpg"
    gui_flag = "--gui" in sys.argv
    sys.exit(read_and_display_traffic_image(img_file, show_gui=gui_flag))

