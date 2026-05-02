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
    base_options=BaseOptions(model_asset_path=".\\hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.3,
    num_hands=2   # allow detection of up to 2 hands
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

DATA_DIR = './isl_dataset/Indian'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Detect hands
        result = hand_landmarker.detect(mp_image)

        # ✅ Accept samples with 1 or 2 hands
        if result.hand_landmarks and len(result.hand_landmarks) in [1, 2]:
            # Collect all x and y values
            for hand_landmarks in result.hand_landmarks:
                for lm in hand_landmarks:
                    x_.append(lm.x)
                    y_.append(lm.y)

            # Normalize and flatten
            for hand_landmarks in result.hand_landmarks:
                for lm in hand_landmarks:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            # If only one hand detected, pad with zeros (84 features)
            if len(result.hand_landmarks) == 1:
                data_aux.extend([0] * 84)

            data.append(data_aux)
            labels.append(dir_)

# Save dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Explicitly close to avoid shutdown error
hand_landmarker.close()
