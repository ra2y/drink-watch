from ultralytics import YOLO


def detect_objects(frame, model):
    # ensure tracking persists
    result = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[
        0
    ]
    objects = []
    
    # print each detected object's class, confidence, bounds, and tracking ID.
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls)
            name = model.names[class_id]
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            track_id = int(box.id) if box.id is not None else None
            print(name, confidence, track_id, (x1, y1, x2, y2))
            objects.append(
                {"bbox": (x1, y1, x2, y2), "track_id": track_id, "class_id": class_id}
            )
    
    return objects, result.plot()


def create_model():
    return YOLO("./models/yolo26n.pt")
