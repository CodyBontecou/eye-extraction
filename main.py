# Install required packages
# !pip install torch torchvision transformers pillow opencv-python face-alignment

import torch
from torchvision import transforms
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import cv2
import face_alignment
import numpy as np

# Load the DETR model for face detection
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Load the face alignment model for facial landmarks
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False
)


# Function to detect faces
def detect_faces(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Keep only face detections
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    bboxes = postprocessed_outputs["boxes"][
        postprocessed_outputs["labels"] == 1
    ]  # 1 is the label for person
    return bboxes.tolist()


# Function to extract eye regions
def extract_eyes(image_path, face_box):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get facial landmarks
    landmarks = fa.get_landmarks(image_rgb)[0]

    # Extract eye landmarks (indices 36-41 for right eye, 42-47 for left eye)
    right_eye = landmarks[36:42]
    left_eye = landmarks[42:48]

    # Calculate eye bounding boxes
    right_eye_box = np.array([right_eye.min(axis=0), right_eye.max(axis=0)]).astype(int)
    left_eye_box = np.array([left_eye.min(axis=0), left_eye.max(axis=0)]).astype(int)

    # Add some padding
    padding = 10
    right_eye_box += np.array([[-padding, -padding], [padding, padding]])
    left_eye_box += np.array([[-padding, -padding], [padding, padding]])

    # Extract eye regions
    right_eye_region = image[
        right_eye_box[0][1] : right_eye_box[1][1],
        right_eye_box[0][0] : right_eye_box[1][0],
    ]
    left_eye_region = image[
        left_eye_box[0][1] : left_eye_box[1][1], left_eye_box[0][0] : left_eye_box[1][0]
    ]

    return right_eye_region, left_eye_region


# Main function to process an image
def extract_eyes_from_image(image_path):
    face_boxes = detect_faces(image_path)
    if not face_boxes:
        print("No faces detected in the image.")
        return None, None

    # Use the first detected face
    face_box = face_boxes[0]
    right_eye, left_eye = extract_eyes(image_path, face_box)

    return right_eye, left_eye


# Example usage
image_path = "images/blink.jpg"
right_eye, left_eye = extract_eyes_from_image(image_path)

if right_eye is not None and left_eye is not None:
    cv2.imwrite("right_eye.jpg", right_eye)
    cv2.imwrite("left_eye.jpg", left_eye)
    print("Eye regions extracted and saved.")
else:
    print("Failed to extract eye regions.")
