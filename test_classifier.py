import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import numpy as np

# Load trained model (must be trained on 84 features!)
model_dict = pickle.load(open("./model.p", 'rb'))
model = model_dict['model']

# Setup HandLandmarker
BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="C:/Users/Lenovo/Desktop/sign language translator/hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.3,
    num_hands=2
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Define hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm connections
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        h, w, _ = frame.shape
        x_, y_, data_aux = [], [], []
        all_points = []  # store all landmark points for bounding box

        # Draw landmarks for all detected hands
        for hand_landmarks in result.hand_landmarks:
            points = []
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
                all_points.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                cv2.line(frame, points[start_idx], points[end_idx], (0, 0, 255), 2)

            # Collect raw coordinates for normalization
            for lm in hand_landmarks:
                x_.append(lm.x)
                y_.append(lm.y)

        # Build feature vector (normalize across both hands)
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        # ✅ Only predict if feature length matches training
        # ✅ Only predict if feature length matches training
        if len(data_aux) == 84:
            prediction = model.predict([data_aux])[0]

            # Draw bounding box around all points
            x_min = min([p[0] for p in all_points])
            y_min = min([p[1] for p in all_points])
            x_max = max([p[0] for p in all_points])
            y_max = max([p[1] for p in all_points])

            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 0, 0), 5)
            cv2.putText(frame, str(prediction), (x_min, y_min - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        else:
            # If unable to predict, show "Trying to predict..."
            if all_points:  # only if we have landmarks
                x_min = min([p[0] for p in all_points])
                y_min = min([p[1] for p in all_points])
                x_max = max([p[0] for p in all_points])
                y_max = max([p[1] for p in all_points])

                cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 0, 0), 5)
                cv2.putText(frame, "Trying to predict...", (x_min, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

    cv2.imshow("Hand Landmarks", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
