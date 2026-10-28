import cv2
import time
import mediapipe as mp
import urllib.request
from pathlib import Path

# Model Path
model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
model_path = Path("./models/hand_landmarker.task")

# Download
if not model_path.exists():
    urllib.request.urlretrieve(model_url, model_path)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

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

# MediaPipe Configuration
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=VisionRunningMode.VIDEO,
    # result_callback=print_result,
    num_hands=100,
)

timpestamp_ms = 0

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("NO WEBCAM LOL!")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("CANT READ FRAME GG FRIKIN EZ!")
            break

        # Flip the frame so its like a mirror
        # frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, 
            data=rgb_frame
        )

        timestamp_ms = int(time.time() * 1000)

        # Submit frame synchronously
        result = landmarker.detect_for_video(
            mp_image, 
            timestamp_ms
        )


        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
            points = []
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )
            for start_index, end_index in CONNECTIONS:
                cv2.line(
                    frame,
                    points[start_index],
                    points[end_index],
                    (255, 0, 0),
                    2
                )

            handedness = result.handedness[hand_index][0]
            label = f"{handedness.category_name}: {handedness.score:.2f}"

            wrist = hand_landmarks[0]
            wrist_x = int(wrist.x * width)
            wrist_y = int(wrist.y * height)

            cv2.putText(
                frame,
                label,
                (wrist_x, wrist_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        # Display Webcam
        cv2.imshow("Hand Tracking", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
