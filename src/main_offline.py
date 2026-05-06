"""
main_offline.py
---------------
Offline UAV-SAR pipeline: process a pre-recorded DJI video + SRT telemetry file,
detect persons frame by frame, geolocate each detection, send a one-time email
alert on first detection, and save all results to CSV.

Usage
-----
    python src/main_offline.py --config config.yaml

    # or override specific fields:
    python src/main_offline.py --video path/to/video.MP4 --srt path/to/video.SRT

Requirements
------------
    - DJI video (.MP4) recorded with 'Video Caption' enabled in the DJI app
    - Matching .SRT telemetry file (same base name as video)
    - YOLOv8 weights (.pt) in models/
    - config.yaml filled in (copy from config.yaml.template)
"""

import argparse
import csv
import os
import sys
import cv2
import yaml

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))
from alert       import send_alert
from detect      import PTDetector as HumanDetector, draw_detections
from geolocation import get_real_coords
from srt_reader  import get_frame_telemetry, load_srt

CSV_FIELDS = [
    "frame_id", "drone_lat", "drone_lon", "altitude",
    "person_idx", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "est_lat", "est_lon", "conf",
]

def parse_args():
    p = argparse.ArgumentParser(description="AlienSight offline pipeline")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--video",  default=None)
    p.add_argument("--srt",    default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()

def load_config(path):
    if not os.path.exists(path):
        print(f"[ERROR] Config file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI Overrides
    video_path  = args.video  or cfg["paths"]["video"]
    srt_path    = args.srt    or cfg["paths"]["srt"]
    output_csv  = args.output or cfg["paths"]["output_csv"]
    model_path  = cfg["paths"]["model"]
    camera      = cfg["camera"]
    alert_cfg   = cfg.get("alert", {})
    
    # Correcting gimbal pitch for math (looking down is usually negative)
    gimbal_pitch = cfg.get("gimbal_pitch", 0)
    pitch_input  = 0 if gimbal_pitch == 0 else -abs(gimbal_pitch)

    # 1. Load Telemetry
    telemetry = load_srt(srt_path)
    if not telemetry:
        sys.exit(1)

    # 2. Load Detector (PTDetector for offline usually works best)
    detector = HumanDetector(model_path, conf_threshold=cfg.get("conf_threshold", 0.6))
    
    # 3. Video Setup
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = camera.get("fps", 2) # Processing at 2fps is usually enough for offline
    frame_skip = max(1, round(actual_fps / target_fps))

    print(f"[System] Processing {video_path} at {target_fps} FPS...")

    frame_id = 0
    processed_count = 0
    alert_sent = False
    rows = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frame_id % frame_skip != 0:
                frame_id += 1
                continue

            # Get telemetry for the current timestamp
            telem = get_frame_telemetry(telemetry, frame_id / actual_fps) 
            if telem is None: break

            # Inference (Let the detector handle internal resizing)
            detections = detector.detect(frame)

            if detections:
                print(f"  Frame {frame_id} | Found {len(detections)} person(s)")
                
                for i, det in enumerate(detections):
                    est_lat, est_lon = get_real_coords(
                        bbox_center    = det["center"],
                        drone_position = (telem["lat"], telem["lon"], telem["altitude"]),
                        drone_heading  = telem["heading"],
                        camera_config  = camera
                    )

                    rows.append({
                        "frame_id"  : frame_id,
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

                # Alerting Logic
                if alert_cfg.get("enabled") and not alert_sent:
                    annotated = draw_detections(frame, detections)
                    send_alert(
                        lat=est_lat, lon=est_lon, 
                        person_count=len(detections),
                        recipient_email=alert_cfg["recipient"],
                        sender_email=alert_cfg["sender"],
                        sender_password=alert_cfg["password"],
                        frame=frame, 
                        annotated_frame=annotated
                    )
                    alert_sent = True

            frame_id += 1
            processed_count += 1

    except KeyboardInterrupt:
        print("\n[Stop] Interrupted by user.")
    finally:
        cap.release()

    # 4. Save Results
    if rows:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Done] {len(rows)} detections saved to {output_csv}")
    else:
        print("[Done] No detections found.")

if __name__ == "__main__":
    main()