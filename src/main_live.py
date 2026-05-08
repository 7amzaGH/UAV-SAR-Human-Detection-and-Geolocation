"""
main_live.py
------------
Live UAV-SAR pipeline for real-time onboard deployment.
Optimized for Qualcomm RB3 Gen 2, Jetson, and other edge devices via ONNX Runtime.

Usage:
    python main_live.py --config config.yaml

Inputs:
    - Live Camera feed (USB/CSI/RTSP)
    - Real-time NMEA Telemetry via Serial (Flight Controller)
    - Optimized ONNX Model Weights

Outputs:
    - Real-time console telemetry logs
    - Automated email alerts with geolocation and annotated frames
    - Mission log file (mission.log)
"""

import cv2
import serial
import threading
import logging
import yaml
import sys
import os
from datetime import datetime

# Import local modules
sys.path.insert(0, os.path.dirname(__file__))
from detect import ONNXDetector, draw_detections
from geolocation import get_real_coords
from alert import send_alert

# ── Setup Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("mission.log"), logging.StreamHandler()]
)
log = logging.getLogger("SAR-NPU")

# ── Load Configuration ────────────────────────────────────────────────────────
def load_config():
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        log.critical(f"Failed to load config.yaml: {e}")
        sys.exit(1)

cfg = load_config()

# ── Global Telemetry State ────────────────────────────────────────────────────
telemetry = {"lat": None, "lon": None, "alt": None, "hdg": 0.0, "ready": False}

def gps_worker():
    """Background thread to handle GPS serial data."""
    import pynmea2
    try:
        ser = serial.Serial(cfg['gps']['port'], cfg['gps']['baud'], timeout=1)
        log.info(f"GPS Serial connected: {cfg['gps']['port']}")
    except Exception as e:
        log.error(f"GPS connection failed: {e}. Running in simulation/offline mode.")
        return

    while True:
        try:
            line = ser.readline().decode("ascii", errors="replace").strip()
            if line.startswith("$GPGGA"):
                msg = pynmea2.parse(line)
                if msg.latitude and msg.longitude:
                    telemetry["lat"], telemetry["lon"] = msg.latitude, msg.longitude
                    telemetry["alt"] = float(msg.altitude)
            elif line.startswith("$GPHDT"):
                telemetry["hdg"] = float(pynmea2.parse(line).heading)
            
            if all(telemetry[k] is not None for k in ["lat", "lon", "alt"]):
                telemetry["ready"] = True
        except Exception:
            continue

# ── Main Mission Loop ─────────────────────────────────────────────────────────
def main():
    # Start GPS Thread
    threading.Thread(target=gps_worker, daemon=True).start()

    # Initialize Detector
    detector = ONNXDetector(
        model_path=cfg['model']['path'],
        provider=cfg['model']['provider'],
        conf_threshold=cfg['model']['threshold']
    )

    # Initialize Camera
    cap = cv2.VideoCapture(cfg['camera']['source'])
    if not cap.isOpened():
        log.error("Could not open camera source.")
        return

    alert_sent = False
    log.info("System Ready. Waiting for GPS fix...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if not telemetry["ready"]:
                # Optional: Show 'Wait' on frame if debugging
                continue

            # 1. Inference
            detections = detector.detect(frame)

            # 2. Geolocation & Logic
            if detections:
                log.info(f"Detected {len(detections)} targets.")
                
                # Geolocate the most confident target
                best_target = max(detections, key=lambda d: d['conf'])
                est_lat, est_lon = get_real_coords(
                    bbox_center=best_target["center"],
                    drone_position=(telemetry["lat"], telemetry["lon"], telemetry["alt"]),
                    drone_heading=telemetry["hdg"],
                    gimbal_pitch   = pitch_input,
                    camera_config=cfg['camera']
                )

                log.info(f"Target Est Location: {est_lat:.6f}, {est_lon:.6f}")

                # 3. Alerting
                if cfg['email']['enabled'] and not alert_sent:
                    annotated = draw_detections(frame, detections)
                    success = send_alert(
                        lat=est_lat, lon=est_lon, 
                        person_count=len(detections),
                        recipient_email=cfg['email']['recipient'],
                        sender_email=cfg['email']['sender'],
                        sender_password=cfg['email']['password'], 
                        annotated_frame=annotated
                    )
                    if success:
                        alert_sent = True
                        log.info("Emergency Alert Sent.")

    except KeyboardInterrupt:
        log.info("Mission interrupted by user.")
    finally:
        cap.release()
        log.info("Mission ended. Resources released.")

if __name__ == "__main__":
    main()
