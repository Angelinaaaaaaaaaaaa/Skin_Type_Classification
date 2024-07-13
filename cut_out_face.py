import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt

def cut_face(from_directory, img_name, threshold=0.5):
    # Initialize the Mediapipe face detection solution
    mp_face_detection = mp.solutions.face_detection

    # Read the input image
    # directory = './'
    # img_name = 'face.jpg'
    image = cv2.imread(from_directory + img_name)

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=threshold) as face_detection:
        results = face_detection.process(rgb_image)

    # Single out the detected face
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                            int(bboxC.width * iw), int(bboxC.height * ih))
            x = max(0, x)
            y = max(0, y)
            x_end = min(x + w, iw)
            y_end = min(y + h, ih)
            cropped_image = image[y: y_end, x: x_end]
            return cropped_image
    else:
        print(f'None detected: {from_directory}{img_name}')
        return None
        
def save_to_file(to_directory, img_name, img):
    cv2.imwrite(to_directory + img_name + '_faceonly.jpg', img)

def get_dimension(img):
    if isinstance(img, np.ndarray):
        return img.shape[0:2]
    else:
        return
    
def draw_frame(from_directory, img_name):
    # Initialize the Mediapipe face detection solution
    mp_face_detection = mp.solutions.face_detection

    # Read the input image
    # directory = './'
    # img_name = 'face.jpg'
    image = cv2.imread(from_directory + img_name)

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(rgb_image)

    # Draw rectangles around detected faces
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                            int(bboxC.width * iw), int(bboxC.height * ih))
            print('DEBUG:', x, y, x + w, y + h)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()

def pad_image(image, target_height, target_width, pad_color=(128, 128, 128)):
    # Get current dimensions
    height, width, channels = image.shape

    # Compute padding sizes
    pad_height = target_height - height
    pad_width = target_width - width

    # Compute padding for top, bottom, left, right
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # Pad the image
    padded_image = cv2.copyMakeBorder(
        image,
        top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )

    return padded_image