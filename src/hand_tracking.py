import cv2
import time
import mediapipe as mp

# Model Path
model_path = "./models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a hand landmarker instance with the live stream mode:
def print_result(
    result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    print("hand landmarker result: {}".format(result))


# MediaPipe Configuration
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=100,
)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("NO WEBCAM LOL!")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            print("CANT READ FRAME GG FRIKIN EZ!")
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Need to know frame timestamps bc frame processing is async
        frame_timestamp_ms = int(time.monotonic() * 1000)

        # Submit frame asynchronously
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Display Webcam
        cv2.imshow("Hand Tracking", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
