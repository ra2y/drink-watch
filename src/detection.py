import cv2
from ultralytics import YOLO

model = YOLO("./models/yolo26n.pt")

# Object Detection
results = model.track(
    source=0, persist=True, tracker="bytetrack.yaml", stream=True, show=True
)

# Label boxes
for r in results:
    if r.boxes is None:
        continue
    for box in r.boxes:
        class_id = int(box.cls)
        name = model.names[class_id]
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(name, confidence, (x1, y1, x2, y2))

cv2.destroyAllWindows()
