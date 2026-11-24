import cv2
import json
import time
import numpy as np
from detection import create_model, detect_objects
from hand_tracking import create_landmarker, detect_hands, draw_hands
from rule_engine import check_rules


def load_zones(filename):
    with open(filename, "r") as file:
        zone_data = json.load(file)
    zones = {}

    for zone_name, points in zone_data.items():
        zones[zone_name] = np.array(points, dtype=np.int32)
    return zones


# Connects hand to people detected by YOLO #YOUONLYLIVEONCE!
def get_hand_owner(wrist_pt, people):
    wrist_x, wrist_y = wrist_pt

    for person in people:
        x1, y1, x2, y2 = person["bbox"]
        if x1 <= wrist_x <= x2 and y1 <= wrist_y <= y2:
            return person["track_id"]
    return None


def main():
    zones = load_zones("config/zones.json")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("DA CAM IS NOT ON!")
        return

    person_model = create_model()
    hand_landmarker = create_landmarker()
    last_timestamp = 0

    while True:
        success, frame = camera.read()
        if not success:
            print("we cannot read dis frame")
            break

        frame_height, frame_width = frame.shape[:2]

        objects, annotated_frame = detect_objects(frame, person_model)
        people = [
            obj
            for obj in objects
            if obj["class_id"] == 0 and obj["track_id"] is not None
        ]
        timestamp = max(int(time.monotonic() * 1000), last_timestamp + 1)
        last_timestamp = timestamp
        hands = detect_hands(frame, hand_landmarker, timestamp)
        frame = annotated_frame
        draw_hands(frame, hands)

        for hand in hands.hand_landmarks:
            # normalize wrist coords
            wrist_x = int(hand[0].x * frame_width)
            wrist_y = int(hand[0].y * frame_height)

            # normalize fingertip coords
            finger_x = int(hand[8].x * frame_width)
            finger_y = int(hand[8].y * frame_height)

            owner_id = get_hand_owner((wrist_x, wrist_y), people)
            if owner_id is not None:
                event = check_rules(
                    track_id=owner_id, fingertip_point=(finger_x, finger_y), zones=zones
                )
                if event is not None:
                    print("EVENT:", event)

                cv2.putText(
                    frame,
                    f"Person {owner_id}",
                    (finger_x, finger_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.circle(frame, (finger_x, finger_y), 6, (0, 0, 255), -1)

        # Draw the configured zones
        for zone_name, polygon in zones.items():
            cv2.polylines(frame, [polygon], True, (255, 0, 0), 2)

            x, y = polygon[0]
            cv2.putText(
                frame,
                zone_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        cv2.imshow("Bar Surveillance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    hand_landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
