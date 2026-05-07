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

sys.path.insert(0, os.path.dirname(__file__))

from alert       import send_alert
from detect      import PTDetector as HumanDetector, draw_detections
from geolocation import get_real_coords
from srt_reader  import get_frame_telemetry, load_srt


# ── CSV columns ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "frame_id", "drone_lat", "drone_lon", "altitude",
    "person_idx", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "est_lat", "est_lon", "conf",
]


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="AlienSight offline pipeline")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--video",  default=None, help="Override config video path")
    p.add_argument("--srt",    default=None, help="Override config SRT path")
    p.add_argument("--output", default=None, help="Override config CSV output path")
    return p.parse_args()


# ── Config loading ────────────────────────────────────────────────────────────
def load_config(path):
    if not os.path.exists(path):
        print(f"[ERROR] Config file not found: {path}")
        print("        Copy config.yaml.template to config.yaml and fill it in.")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # Allow CLI to override config paths
    video_path = args.video  or cfg["paths"]["video"]
    srt_path   = args.srt    or cfg["paths"]["srt"]
    output_csv = args.output or cfg["paths"]["output_csv"]
    model_path = cfg["paths"]["model"]

    camera       = cfg["camera"]
    gimbal_pitch = cfg.get("gimbal_pitch", 0)
    alert_cfg    = cfg.get("alert", {})

    # gimbal_pitch in config is a positive number (e.g. 45 for oblique).
    # get_real_coords expects 0 for nadir, negative for below-horizon oblique.
    pitch_input = 0 if gimbal_pitch == 0 else -abs(gimbal_pitch)

    # ── Load telemetry ────────────────────────────────────────────────────────
    telemetry = load_srt(srt_path)
    if not telemetry:
        sys.exit(1)
    print(f"[SRT]   Loaded {len(telemetry)} frames from {srt_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    detector = HumanDetector(model_path, conf_threshold=cfg.get("conf_threshold", 0.6))

    # ── Video setup ───────────────────────────────────────────────────────────
    cap        = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = camera.get("fps", 2)
    frame_skip = max(1, round(actual_fps / target_fps))

    print(f"[Video] {actual_fps:.0f} fps actual -> processing every {frame_skip} frame(s) "
          f"({actual_fps / frame_skip:.0f} fps effective)")

    # ── Main loop ─────────────────────────────────────────────────────────────
    alert_sent   = False
    rows         = []
    raw_frame_id = 0   # every frame read from the video
    proc_frame_id = 0  # only frames we actually run inference on

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames we don't want to process
        if raw_frame_id % frame_skip != 0:
            raw_frame_id += 1
            continue

        # Map processed-frame index -> SRT entry (1 SRT entry per second of video)
        telem = get_frame_telemetry(telemetry, proc_frame_id)
        if telem is None:
            break

        drone_pos = (telem["lat"], telem["lon"], telem["altitude"])

        # ── Inference ─────────────────────────────────────────────────────────
        detections = detector.detect(frame)

        if detections:
            n = len(detections)
            print(f"  Frame {proc_frame_id:04d} | "
                  f"{telem['lat']:.6f}, {telem['lon']:.6f} | "
                  f"Alt {telem['altitude']:.1f} m | "
                  f"{n} person(s)")

            for i, det in enumerate(detections):
                est_lat, est_lon = get_real_coords(
                    bbox_center    = det["center"],
                    drone_position = drone_pos,
                    drone_heading  = telem["heading"],
                    gimbal_pitch   = pitch_input,
                    camera_config  = camera,
                )
                print(f"    └─ Person {i+1} (conf {det['conf']:.2f}) "
                      f"-> {est_lat:.6f}, {est_lon:.6f}")

                rows.append({
                    "frame_id"  : proc_frame_id,
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

            # ── One email on first detection ───────────────────────────────
            if alert_cfg.get("enabled") and not alert_sent:
                best = max(detections, key=lambda d: d["conf"])
                best_lat, best_lon = get_real_coords(
                    bbox_center    = best["center"],
                    drone_position = drone_pos,
                    drone_heading  = telem["heading"],
                    gimbal_pitch   = pitch_input,
                    camera_config  = camera,
                )
                annotated = draw_detections(frame, detections)
                alert_sent = send_alert(
                    lat             = best_lat,
                    lon             = best_lon,
                    person_count    = n,
                    recipient_email = alert_cfg["recipient"],
                    sender_email    = alert_cfg["sender"],
                    sender_password = alert_cfg["password"],
                    annotated_frame = annotated,
                )

        raw_frame_id  += 1
        proc_frame_id += 1

    cap.release()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Done] {proc_frame_id} frames processed")
    print(f"[Done] {len(rows)} detections -> {output_csv}")
    if not alert_sent and alert_cfg.get("enabled"):
        print("[Done] No persons detected — no alert sent")


if __name__ == "__main__":
    main()
