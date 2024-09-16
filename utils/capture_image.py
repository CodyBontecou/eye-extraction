import cv2
import time
from datetime import datetime


def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    time.sleep(2)  # Wait for the camera to initialize

    for _ in range(10):
        ret, frame = cap.read()
        if ret and not is_image_black(frame):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"webcam_capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image captured: {filename}")
            cap.release()
            return filename
        time.sleep(0.1)

    print("Error: Could not capture a non-black image after multiple attempts.")
    cap.release()
    return None


def is_image_black(image):
    # Check if the image is mostly black
    return cv2.mean(image)[0] < 5  # Adjust this threshold if needed
