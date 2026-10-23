import cv2
from ultralytics import YOLO

model = YOLO("./models/yolov8n.pt")

results = model.track(
    source=0,
    persist=True,
    tracker="bytetrack.yaml",
    stream=True,
    show=True
)

for r in results:
    if r.boxes is not None:
        confidences = r.boxes.conf.cpu().numpy()

        boxes = r.boxes.xyxy.cpu().numpy()
        if r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
        else:
            track_ids = []
        class_ids = r.boxes.cls.int().cpu().tolist()
        for class_id in class_ids:
                class_name = model.names[class_id]
                print(f"Detected: {class_name} (Class ID: {class_id})")

for r in results:

    annotated_frame = r.plot()
    
    cv2.imshow("YOLO Tracking", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()