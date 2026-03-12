import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import cv2

def get_city(lat, lon):
    """Reverse geocode lat/lon to a city name using Nominatim (free, no API key)."""
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="uav_sar_alert")
        location = geolocator.reverse(f"{lat}, {lon}", timeout=5)
        address  = location.raw.get("address", {})
        city     = address.get("city") or address.get("town") or address.get("village") or "Unknown"
        country  = address.get("country", "")
        return f"{city}, {country}"
    except Exception:
        return "Unknown"

def send_alert(lat, lon, person_count, recipient_email, sender_email, sender_password, frame=None, annotated_frame=None):
    """
    Send an email alert with:
      - Number of persons detected
      - Location (lat, lon, city)
      - Google Maps link
      - Timestamp
      - Raw frame (no bounding box)
      - Annotated frame (with bounding box)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    maps_link = f"https://maps.google.com/?q={lat},{lon}"
    city      = get_city(lat, lon)

    label   = "Person" if person_count == 1 else "Persons"
    subject = f"Drone Alert: {person_count} {label} Detected"
    body    = f"""{person_count} {label} detected!

Location:
  Latitude:  {lat:.4f}
  Longitude: {lon:.4f}
  City: {city}
  Maps: {maps_link}

Time: {timestamp}
    """

    msg = MIMEMultipart()
    msg["From"]    = sender_email
    msg["To"]      = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach raw frame
    if frame is not None:
        _, img_encoded = cv2.imencode(".jpg", frame)
        raw_attachment = MIMEImage(img_encoded.tobytes(), name="raw_frame.jpg")
        raw_attachment.add_header("Content-Disposition", "attachment", filename="raw_frame.jpg")
        msg.attach(raw_attachment)

    # Attach annotated frame (with bounding box)
    if annotated_frame is not None:
        _, img_encoded = cv2.imencode(".jpg", annotated_frame)
        ann_attachment = MIMEImage(img_encoded.tobytes(), name="detection.jpg")
        ann_attachment.add_header("Content-Disposition", "attachment", filename="detection.jpg")
        msg.attach(ann_attachment)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print(f"Alert sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send alert: {e}")
