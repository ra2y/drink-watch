import cv2
from ultralytics import YOLO

# Download YOLO model
model = YOLO("./models/yolo26n.pt")

# YOLO Object Detection
results = model.track(
    source=0, persist=True, tracker="bytetrack.yaml", stream=True, show=False
)

# Process each frame
for r in results:
    # Process detected objects
    if r.boxes is not None:
        for box in r.boxes:
            # Class Info
            class_id = int(box.cls)
            name = model.names[class_id]
            # Confidence Score
            confidence = float(box.conf)
            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Tracking ID
            if box.id is not None:
                track_id = int(box.id)
            else:
                track_id = None
            # Print to terminal
            print(name, confidence, track_id, (x1, y1, x2, y2))

    # Draw bounding boxes, labels, and tracking information
    annotated_frame = r.plot()

    # Display annotated video
    cv2.imshow("YOLO Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
