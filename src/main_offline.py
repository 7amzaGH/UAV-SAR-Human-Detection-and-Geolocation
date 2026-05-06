"""
main_offline.py
---------------
Offline UAV-SAR pipeline: processes a pre-recorded DJI video + SRT file,
detects humans frame by frame, geolocates each detection, sends a one-time
email alert on first detection, and saves all results to a CSV file.

Usage:
    python src/main_offline.py

Requirements:
    - DJI video file (.MP4)
    - DJI SRT telemetry file (.SRT) — same name as video, enable Video Caption
      in DJI camera settings before recording
    - YOLOv8 model weights (.pt)

Output:
    - detections.csv  — all detections with bbox, estimated GPS, confidence
"""

import cv2
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(__file__))

from detect      import HumanDetector
from geolocation import get_real_coords
from alert       import send_alert
from srt_reader  import load_srt, get_frame_telemetry

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_PATH = "models/best.pt"
VIDEO_PATH = "DJI_0001.MP4"
SRT_PATH   = "DJI_0001.SRT"
OUTPUT_CSV = "detections.csv"

CAMERA = {
    "fov_h":        76,    # horizontal field of view (degrees) — DJI Air 3S
    "fov_v":        49,    # vertical field of view (degrees)
    "image_width":  1920,  # original frame resolution
    "image_height": 1080,
    "fps":          30,    # target processing fps — set to match your recording
                           # DJI Air 3S common values: 24, 30, 60
                           # check actual fps printed at startup and adjust
}

# Gimbal pitch — not stored in DJI SRT, set manually per flight
# 0  = nadir (straight down)
# 45 = oblique (45 degrees below horizon)
GIMBAL_PITCH = 0

# Email alert — set ALERT_EMAIL to "" to disable
ALERT_EMAIL     = "rescue_team@example.com"
SENDER_EMAIL    = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"   # Gmail App Password, not account password

# ── Load Telemetry ─────────────────────────────────────────────────────────────
telemetry = load_srt(SRT_PATH)
if not telemetry:
    print(f"[ERROR] No telemetry found in {SRT_PATH}")
    print("        Make sure 'Video Caption' is ON in DJI camera settings.")
    sys.exit(1)

print(f"[SRT]   Loaded {len(telemetry)} telemetry frames from {SRT_PATH}")

# ── Load Model ─────────────────────────────────────────────────────────────────
detector    = HumanDetector(MODEL_PATH, conf_threshold=0.6)
alert_sent  = False
pitch_input = 0 if GIMBAL_PITCH == 0 else -GIMBAL_PITCH
rows        = []

# ── Video Setup ────────────────────────────────────────────────────────────────
cap        = cv2.VideoCapture(VIDEO_PATH)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
target_fps = CAMERA["fps"]
frame_skip = max(1, round(actual_fps / target_fps))

print(f"[Video] Actual FPS: {actual_fps:.1f} | "
      f"Target FPS: {target_fps} | "
      f"Processing every {frame_skip} frame(s)")

# ── Main Loop ──────────────────────────────────────────────────────────────────
frame_id     = 0   # raw frame counter from video
processed_id = 0   # logical frame index after skip

while cap.isOpened():
    ret, frame_orig = cap.read()
    if not ret:
        break

    # Skip frames to match target fps
    if frame_id % frame_skip != 0:
        frame_id += 1
        continue

    telem = get_frame_telemetry(telemetry, processed_id)
    if telem is None:
        break

    drone_pos = (telem["lat"], telem["lon"], telem["altitude"])

    # Resize to 960×960 for YOLO inference, scale bbox back to original resolution
    frame_yolo = cv2.resize(frame_orig, (960, 960))
    detections = detector.detect(frame_yolo)

    scale_x = frame_orig.shape[1] / 960
    scale_y = frame_orig.shape[0] / 960
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        det["bbox"]   = (int(x1 * scale_x), int(y1 * scale_y),
                         int(x2 * scale_x), int(y2 * scale_y))
        det["center"] = (int((x1 + x2) / 2 * scale_x),
                         int((y1 + y2) / 2 * scale_y))

    if detections:
        print(f"Frame {processed_id:04d} | "
              f"Drone: {telem['lat']:.6f}, {telem['lon']:.6f} | "
              f"Alt: {telem['altitude']:.1f} m | "
              f"{len(detections)} person(s)")

        for i, det in enumerate(detections):
            est_lat, est_lon = get_real_coords(
                bbox_center    = det["center"],
                drone_position = drone_pos,
                drone_heading  = telem["heading"],
                gimbal_pitch   = pitch_input,
                camera_config  = CAMERA,
            )
            print(f"  └─ Person {i+1} (conf {det['conf']:.2f}) → "
                  f"Est: {est_lat:.6f}, {est_lon:.6f}")

            rows.append({
                "frame_id"  : processed_id,
                "drone_lat" : telem["lat"],
                "drone_lon" : telem["lon"],
                "altitude"  : telem["altitude"],
                "person_idx": i + 1,
                "bbox_x1"   : det["bbox"][0],
                "bbox_y1"   : det["bbox"][1],
                "bbox_x2"   : det["bbox"][2],
                "bbox_y2"   : det["bbox"][3],
                "est_lat"   : round(est_lat, 6),
                "est_lon"   : round(est_lon, 6),
                "conf"      : det["conf"],
            })

        # Send one email on first detection with annotated frame attached
        if ALERT_EMAIL and not alert_sent:
            best = max(detections, key=lambda d: d["conf"])
            est_lat, est_lon = get_real_coords(
                bbox_center    = best["center"],
                drone_position = drone_pos,
                drone_heading  = telem["heading"],
                gimbal_pitch   = pitch_input,
                camera_config  = CAMERA,
            )
            success = send_alert(
                lat             = est_lat,
                lon             = est_lon,
                person_count    = len(detections),
                recipient_email = ALERT_EMAIL,
                sender_email    = SENDER_EMAIL,
                sender_password = SENDER_PASSWORD,
                frame           = frame_orig,
                annotated_frame = detector.draw(frame_orig, detections),
            )
            if success:
                alert_sent = True

    frame_id     += 1
    processed_id += 1

cap.release()

# ── Save CSV ───────────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="") as f:
    fieldnames = ["frame_id", "drone_lat", "drone_lon", "altitude",
                  "person_idx", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                  "est_lat", "est_lon", "conf"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[Done] Processed {processed_id} frames at {target_fps} fps")
print(f"[Done] {len(rows)} detections saved → {OUTPUT_CSV}")
if not alert_sent and ALERT_EMAIL:
    print("[Done] No persons detected — no alert sent")
