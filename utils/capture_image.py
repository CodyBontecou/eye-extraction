import cv2
import time
from datetime import datetime


def capture_image():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Wait for the camera to initialize and adjust light levels
    time.sleep(2)

    # Capture multiple frames
    for i in range(10):
        ret, frame = cap.read()
        if ret and not is_image_black(frame):
            # Generate a filename with current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"webcam_capture_{timestamp}.jpg"

            # Save the captured frame
            cv2.imwrite(filename, frame)
            print(f"Image captured: {filename}")
            break
        time.sleep(0.1)
    else:
        print("Error: Could not capture a non-black image after multiple attempts.")

    # Release the webcam
    cap.release()


def is_image_black(image):
    # Check if the image is mostly black
    return cv2.mean(image)[0] < 5  # Adjust this threshold if needed
