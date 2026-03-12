import cv2
import serial          # reads GPS from drone flight controller via UART
import pynmea2         # parses NMEA GPS sentences
from detect import HumanDetector
from geolocation import get_real_coords
from alert import send_alert

# ── Settings ──────────────────────────────────────────────────────────────────

MODEL_PATH = "models/best.pt"

# DJI Air 3S camera settings
CAMERA = {
    "fov_h":        84,
    "fov_v":        54,
    "image_width":  1920,
    "image_height": 1080,
}

# Email alert
ALERT_EMAIL     = "rescue_team@example.com"
SENDER_EMAIL    = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"

# GPS serial port on Jetson Nano (check yours with: ls /dev/ttyTHS*)
GPS_PORT = "/dev/ttyTHS1"
GPS_BAUD = 9600

# Fixed values — update if your setup reads these from flight controller too
ALTITUDE = 30   # meters
HEADING  = 0    # degrees

# ── GPS Reader ────────────────────────────────────────────────────────────────

def read_gps(ser):
    """
    Read one GPS fix from the serial port.
    Returns (lat, lon) or None if no valid fix yet.
    """
    try:
        line = ser.readline().decode("ascii", errors="replace")
        if line.startswith("$GPRMC") or line.startswith("$GPGGA"):
            msg = pynmea2.parse(line)
            return msg.latitude, msg.longitude
    except Exception:
        pass
    return None

# ── Run ───────────────────────────────────────────────────────────────────────

detector   = HumanDetector(MODEL_PATH, conf_threshold=0.4)
alert_sent = False

# Connect to camera — Jetson Nano CSI camera via GStreamer
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# Connect to GPS
gps_serial  = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
drone_lat   = 0.0
drone_lon   = 0.0

print("Starting onboard detection — press Ctrl+C to stop")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    # Update GPS position every frame if a fix is available
    fix = read_gps(gps_serial)
    if fix:
        drone_lat, drone_lon = fix

    detections = detector.detect(frame)
    annotated  = detector.draw(frame.copy(), detections)

    if detections and drone_lat != 0.0:
        lat, lon = get_real_coords(
            bbox_center    = detections[0]["center"],
            drone_position = (drone_lat, drone_lon, ALTITUDE),
            drone_heading  = HEADING,
            camera_config  = CAMERA,
        )
        print(f"{len(detections)} person(s) at: {lat:.6f}, {lon:.6f}")

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

    # No imshow on Jetson — no screen attached, would crash
    # cv2.imshow("UAV Detection", annotated)  ← commented out on purpose

cap.release()
gps_serial.close()
