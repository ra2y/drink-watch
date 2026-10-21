# Run

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p config
python src/calibrate_zones.py
```

## Roadmap

- [x] Set up the environment and camera zone calibration
- [ ] Add person detection and tracking with YOLOv8 and ByteTrack
- [ ] Add hand tracking with MediaPipe Hands
- [ ] Detect suspicious activity using zones and dwell time
- [ ] Log events in SQLite and save snapshots or video clips
- [ ] Build a Streamlit review dashboard
- [ ] Test the full system and prepare the final demo and write-up
