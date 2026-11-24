import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path("models/hand_landmarker.task")
CONNECTIONS = [
    (0, 1),  # Thumb
    (1, 2),  # |
    (2, 3),  # |
    (3, 4),  # Thumb
    (0, 5),  # Index
    (5, 6),  # |
    (6, 7),  # |
    (7, 8),  # Index
    (5, 9),  # Middle
    (9, 10),  # |
    (10, 11), # |
    (11, 12),  # Middle
    (9, 13),  # Ring
    (13, 14), # |
    (14, 15), # |
    (15, 16),  # Ring
    (13, 17),  # Pinky
    (17, 18),  # |
    (18, 19),  # |
    (19, 20),  # Pinky
    (0, 17),  # Palm
]


def create_landmarker():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=100,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_hands(frame, landmarker, timestamp):
    # convert BGR to RGB (opencv uses bgr, mediapipe uses rgb)
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    return landmarker.detect_for_video(image, timestamp)


def draw_hands(frame, result):
    height, width = frame.shape[:2]
    for hand_index, landmarks in enumerate(result.hand_landmarks):
        # normalize landmarks to frame dimensions
        points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
        for point in points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
        for start, end in CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 0, 0), 2)
        handedness = result.handedness[hand_index][0]
        label = f"{handedness.category_name}: {handedness.score:.2f}"
        cv2.putText(frame, label, (points[0][0], points[0][1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
