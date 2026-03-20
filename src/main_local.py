import cv2
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from detect import HumanDetector
from geolocation import get_real_coords
from alert import send_alert
from srt_reader import load_srt, get_frame_telemetry

# ── Settings ──────────────────────────────────────────────────────────────────

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "DJI_0001.MP4"
SRT_PATH   = "DJI_0001.SRT"   # same name as video, .SRT extension
                                # enable Video Caption in DJI camera settings

# DJI Air 3S camera — FOV derived from 84° diagonal, 16:9 sensor
CAMERA = {
    "fov_h":        76,
    "fov_v":        49,
    "image_width":  1920,
    "image_height": 1080,
}

# Email alert — leave empty string to disable
ALERT_EMAIL     = "rescue_team@example.com"
SENDER_EMAIL    = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"   # Gmail App Password

# ── Run ───────────────────────────────────────────────────────────────────────

detector   = HumanDetector(MODEL_PATH, conf_threshold=0.6)
alert_sent = False

# Load all telemetry from SRT file upfront
# Per-frame: lat, lon, rel_alt (altitude above ground), heading (defaults to 0)
telemetry = load_srt(SRT_PATH)
if not telemetry:
    print(f"[ERROR] No telemetry found in {SRT_PATH}")
    print("        Make sure Video Caption is ON in DJI camera settings.")
    sys.exit(1)
print(f"[SRT] Loaded {len(telemetry)} telemetry frames from {SRT_PATH}")

cap      = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get GPS, altitude, heading for this exact frame
    telem = get_frame_telemetry(telemetry, frame_id)
    if telem is None:
        break

    drone_pos = (telem["lat"], telem["lon"], telem["altitude"])
    heading   = telem["heading"]

    detections = detector.detect(frame)
    annotated  = detector.draw(frame.copy(), detections)

    if detections:
        # Use detection with highest confidence for geolocation
        best = max(detections, key=lambda d: d["conf"])
        lat, lon = get_real_coords(
            bbox_center    = best["center"],
            drone_position = drone_pos,
            drone_heading  = heading,
            camera_config  = CAMERA,
        )
        print(f"Frame {frame_id:04d} | "
              f"Drone: {telem['lat']:.4f},{telem['lon']:.4f} "
              f"Alt:{telem['altitude']:.1f}m | "
              f"{len(detections)} person(s) at: {lat:.6f},{lon:.6f}")

        if ALERT_EMAIL and not alert_sent:
            send_alert(
                lat             = lat,
                lon             = lon,
                person_count    = len(detections),
                recipient_email = ALERT_EMAIL,
                sender_email    = SENDER_EMAIL,
                sender_password = SENDER_PASSWORD,
                frame           = frame,
                annotated_frame = annotated,
            )
            alert_sent = True

    cv2.imshow("UAV Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
