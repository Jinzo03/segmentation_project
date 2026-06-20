import os
import cv2

TRACKED_DIR = "tracked_output"
VIDEO_OUTPUT_NAME = "cell_tracking_lifecycle.mp4"
FRAMES_PER_SECOND = 6  # 6 FPS makes the slow cell drift easy to observe clearly

def compile_frames_to_video():
    # 1. Gather and sort the tracked image timeline
    frames = sorted([f for f in os.listdir(TRACKED_DIR) if f.lower().endswith(".png")])
    if not frames:
        print(f"Error: No processed frames found in '{TRACKED_DIR}'. Run track_cells.py first!")
        return

    print(f"Found {len(frames)} frames. Preparing video canvas...")
    
    # 2. Inspect the first frame to automatically detect height and width dimensions
    sample_img = cv2.imread(os.path.join(TRACKED_DIR, frames[0]))
    height, width, channels = sample_img.shape

    # 3. Initialize the OpenCV VideoWriter structure
    # 'mp4v' is a universally supported, web-friendly MPEG-4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT_NAME, fourcc, FRAMES_PER_SECOND, (width, height))

    ## 4. Pipe each individual frame sequentially into the video stream buffer
    print("Compiling video file sequence...")
    for frame_name in frames:
        frame_path = os.path.join(TRACKED_DIR, frame_name)
        img = cv2.imread(frame_path)
        video_writer.write(img)

    # 5. Close files and release handle locks
    video_writer.release()
    # cv2.destroyAllWindows()  <-- REMOVE OR COMMENT OUT THIS LINE
    
    print(f"\nSuccess! Video compiled and saved as: '{VIDEO_OUTPUT_NAME}'")

if __name__ == "__main__":
    compile_frames_to_video()