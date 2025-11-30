import glob
import cv2
from ultralytics import YOLO

# Import from the stereo_3d_tracker package
from src.stereo_3d_tracker import (
    MultiObject3DTracker,
    DT,
    create_stereo_matcher,
)

# Main function to run the 3D object tracking
def main():
    model = YOLO("yolo11l.pt") # Load a pre-trained YOLO model
    stereo = create_stereo_matcher() # Create stereo matcher for depth estimation
    tracker = MultiObject3DTracker(model, stereo) # Initialize the 3D tracker

    # Define the sequence and file patterns
    sequence = "seq_01" 
    left_pattern = f"rect_images/{sequence}/left_camera/*.png"
    right_pattern = f"rect_images/{sequence}/right_camera/*.png"

    left_files = sorted(glob.glob(left_pattern))
    right_files = sorted(glob.glob(right_pattern))

    # Create a window to display the results
    window_name = "3D Tracking View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Process each frame pair
    for frame_idx, (left_path, right_path) in enumerate(zip(left_files, right_files)):
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        if left_img is None or right_img is None:
            continue

        stacked = tracker.process_frame(left_img, right_img, frame_idx, dt_frame=DT) # Process the frame pair and get visualization
        cv2.imshow(window_name, stacked) # Display the result
        cv2.resizeWindow(window_name, stacked.shape[1], stacked.shape[0]) # Resize window to fit the image

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
