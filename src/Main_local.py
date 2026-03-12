import cv2
from detect import HumanDetector
from geolocation import get_real_coords
from alert import send_alert

# ── Settings ──────────────────────────────────────────────────────────────────

VIDEO_PATH   = "test_video.mp4"
MODEL_PATH   = "models/best.pt"

# Drone telemetry (update these for each flight)
DRONE_LAT    = 50.264890
DRONE_LON    = 19.023780
ALTITUDE     = 30        # meters
HEADING      = 0         # degrees, 0 = North

# DJI Air 3S camera settings
CAMERA = {
    "fov_h":        84,
    "fov_v":        54,
    "image_width":  1920,
    "image_height": 1080,
}

# Email alert (leave empty strings to disable)
ALERT_EMAIL     = "rescue_team@example.com"
SENDER_EMAIL    = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"

# ── Run ───────────────────────────────────────────────────────────────────────

detector   = HumanDetector(MODEL_PATH, conf_threshold=0.4)
drone_pos  = (DRONE_LAT, DRONE_LON, ALTITUDE)
alert_sent = False

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    annotated  = detector.draw(frame.copy(), detections)

    if detections:
        # Use the first detection's GPS as the alert location
        lat, lon = get_real_coords(
            bbox_center    = detections[0]["center"],
            drone_position = drone_pos,
            drone_heading  = HEADING,
            camera_config  = CAMERA,
        )
        print(f"{len(detections)} person(s) at: {lat:.6f}, {lon:.6f}")

        # Send alert once per run
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

cap.release()
cv2.destroyAllWindows()
