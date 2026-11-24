import time
import cv2

# How long a hand must remain in a zone
DWELL_THRESHOLD = 3.0

# Stores when each person entered a zone
zone_entries = {}

# Prevents repeated events every frame
triggered_tracks = set()


# Return the name of the zone with da point. Return None if da point is outside all zones.
def find_zone(point, zones):
    for zone_name, polygon in zones.items():
        result = cv2.pointPolygonTest(polygon, point, False)

        if result >= 0:
            return zone_name

    return None


# Check if a person's fingertip has stayed inside a zone for too long. (aka they're sus) 
# Returns an event dictionary or None.
def check_rules(track_id, fingertip_point, zones):
    current_time = time.time()

    zone_name = find_zone(fingertip_point, zones)

    # Hand is outside all zones
    if zone_name is None:
        zone_entries.pop(track_id, None)
        triggered_tracks.discard(track_id)
        return None

    # Person just entered a zone
    if track_id not in zone_entries:
        zone_entries[track_id] = {"zone": zone_name, "entered_at": current_time}

        return None

    entry = zone_entries[track_id]

    # Person moved from one zone to another
    if entry["zone"] != zone_name:
        zone_entries[track_id] = {"zone": zone_name, "entered_at": current_time}

        triggered_tracks.discard(track_id)
        return None

    # Calculate how long the hand has been in the zone
    duration = current_time - entry["entered_at"]

    print(f"Track {track_id} has been in " f"{zone_name} for {duration:.1f} seconds")

    # Trigger only once per zone entry
    if duration >= DWELL_THRESHOLD:
        if track_id not in triggered_tracks:
            triggered_tracks.add(track_id)

            return {
                "event_type": "zone_intrusion",
                "zone": zone_name,
                "track_id": track_id,
                "duration": duration,
            }

    return None
