import cv2
import serial
import pynmea2
import threading
from detect import HumanDetector
from geolocation import get_real_coords
from alert import send_alert

# ── Settings ──────────────────────────────────────────────────────────────────

MODEL_PATH = "models/best.pt"

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

# GPS serial port — run: ls /dev/ttyTHS* to find yours
GPS_PORT = "/dev/ttyTHS1"
GPS_BAUD = 9600

# ── Shared telemetry state ────────────────────────────────────────────────────
# Updated continuously by the GPS thread while detection runs in main thread

telemetry = {
    "lat":      None,
    "lon":      None,
    "altitude": None,   # from $GPGGA — meters above sea level
    "heading":  None,   # from $GPHDT or $GPVTG
    "ready":    False,  # True once all four values are available
}

# ── GPS background thread ─────────────────────────────────────────────────────

def read_telemetry(port, baud):
    """
    Runs in background. Reads NMEA sentences from flight controller
    and updates shared telemetry dict in real time.

    Sentences used:
        $GPGGA — lat, lon, altitude
        $GPHDT — heading (true north)
        $GPVTG — heading fallback (course over ground)
    """
    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"[GPS] Connected on {port}")
    except Exception as e:
        print(f"[GPS] Could not open {port}: {e}")
        return

    while True:
        try:
            line = ser.readline().decode("ascii", errors="replace").strip()

            if line.startswith("$GPGGA"):
                msg = pynmea2.parse(line)
                telemetry["lat"]      = msg.latitude
                telemetry["lon"]      = msg.longitude
                telemetry["altitude"] = float(msg.altitude)

            elif line.startswith("$GPHDT"):
                msg = pynmea2.parse(line)
                telemetry["heading"] = float(msg.heading)

            elif line.startswith("$GPVTG") and telemetry["heading"] is None:
                msg = pynmea2.parse(line)
                if msg.true_track:
                    telemetry["heading"] = float(msg.true_track)

            if all(v is not None for v in [
                telemetry["lat"], telemetry["lon"],
                telemetry["altitude"], telemetry["heading"]
            ]):
                telemetry["ready"] = True

        except Exception:
            pass

# ── Start GPS thread ──────────────────────────────────────────────────────────

gps_thread = threading.Thread(
    target=read_telemetry,
    args=(GPS_PORT, GPS_BAUD),
    daemon=True
)
gps_thread.start()

# ── Camera (Jetson Nano CSI via GStreamer) ────────────────────────────────────

gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# ── Run ───────────────────────────────────────────────────────────────────────

detector   = HumanDetector(MODEL_PATH, conf_threshold=0.6)
alert_sent = False

print("[INFO] Waiting for GPS fix...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[Camera] Read failed")
        break

    if not telemetry["ready"]:
        print("[GPS] Waiting for fix...", end="\r")
        continue

    # Snapshot current telemetry
    lat      = telemetry["lat"]
    lon      = telemetry["lon"]
    altitude = telemetry["altitude"]
    heading  = telemetry["heading"]

    detections = detector.detect(frame)
    annotated  = detector.draw(frame.copy(), detections)

    if detections:
        # Use detection with highest confidence for geolocation
        best = max(detections, key=lambda d: d["conf"])
        person_lat, person_lon = get_real_coords(
            bbox_center    = best["center"],
            drone_position = (lat, lon, altitude),
            drone_heading  = heading,
            camera_config  = CAMERA,
        )
        print(f"{len(detections)} person(s) | "
              f"Drone: {lat:.4f},{lon:.4f} "
              f"Alt:{altitude:.1f}m Head:{heading:.1f}° | "
              f"Person at: {person_lat:.6f},{person_lon:.6f}")

        if ALERT_EMAIL and not alert_sent:
            send_alert(
                lat             = person_lat,
                lon             = person_lon,
                person_count    = len(detections),
                recipient_email = ALERT_EMAIL,
                sender_email    = SENDER_EMAIL,
                sender_password = SENDER_PASSWORD,
                frame           = frame,
                annotated_frame = annotated,
            )
            alert_sent = True

    # No imshow on Jetson — no screen during flight
    # Uncomment only when debugging with monitor plugged in:
    # cv2.imshow("UAV Detection", annotated)
    # if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
