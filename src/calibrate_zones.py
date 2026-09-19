import cv2
import numpy as np
import json

cap = cv2.VideoCapture(0)

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8:
        points.append((x,y))
    
cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", mouse_callback)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read from camera")
        break

    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
    if len(points) >= 4:
        till_polygon_points = np.array(points[:4], dtype=np.int32)
        cv2.polylines(
            frame,
            [till_polygon_points],
            isClosed=True,
            color=(0,255,0),
            thickness=3
        )
    if len(points) >= 8:
        drink_polygon_points = np.array(points[4:], dtype=np.int32)
        cv2.polylines(
            frame,
            [drink_polygon_points],
            isClosed=True,
            color=(0,255,0),
            thickness=3
        )    
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        with open("config/zones.json", "w") as file:
            json.dump({"till_zone": points[:4],
                       "drink_zone": points[4:8]
                       }, file, indent=4)
        break




cap.release()
cv2.destroyAllWindows()