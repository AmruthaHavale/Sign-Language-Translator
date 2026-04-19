import os
import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup HandLandmarker
BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="C:/Users/Lenovo/Desktop/sign language translator/hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.3,
    num_hands=2   # ✅ explicitly allow detection of up to 2 hands
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

DATA_DIR = './data'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Detect hands
        result = hand_landmarker.detect(mp_image)

        # ✅ Only save samples when both hands are detected
        if result.hand_landmarks and len(result.hand_landmarks) == 2:
            # Collect all x and y values across both hands
            for hand_landmarks in result.hand_landmarks:
                for lm in hand_landmarks:
                    x_.append(lm.x)
                    y_.append(lm.y)

            # Normalize and flatten both hands into one vector (168 features)
            for hand_landmarks in result.hand_landmarks:
                for lm in hand_landmarks:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            # Append to dataset
            data.append(data_aux)
            labels.append(dir_)

# Save dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
