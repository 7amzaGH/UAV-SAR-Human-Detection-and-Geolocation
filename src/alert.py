"""
alert.py
--------
Sends an email alert when a person is detected, including:
  - Estimated GPS coordinates + Google Maps link
  - Reverse-geocoded city name
  - Annotated detection frame as attachment

Requires a Gmail account with an App Password:
https://support.google.com/accounts/answer/185833
"""

import ssl
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from datetime import datetime

import cv2


def _get_city(lat, lon):
    """Reverse-geocode coordinates to a city string. Returns 'Unknown' on failure."""
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="aliensight_sar")
        location = geolocator.reverse(f"{lat}, {lon}", timeout=5)
        addr = location.raw.get("address", {})
        city = addr.get("city") or addr.get("town") or addr.get("village") or "Unknown"
        return f"{city}, {addr.get('country', '')}".strip(", ")
    except Exception:
        return "Unknown"


def send_alert(lat, lon, person_count, recipient_email,
               sender_email, sender_password, annotated_frame=None):
    """
    Send a detection alert by email.

    Parameters
    ----------
    lat, lon        : float — estimated GPS of the detected person(s)
    person_count    : int   — number of persons detected
    recipient_email : str   — destination address
    sender_email    : str   — Gmail sender address
    sender_password : str   — Gmail App Password
    annotated_frame : ndarray or None — BGR frame with bounding boxes drawn

    Returns
    -------
    bool — True if the email was sent successfully, False otherwise
    """
    label     = "Person" if person_count == 1 else "Persons"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    maps_link = f"https://maps.google.com/?q={lat},{lon}"
    city      = _get_city(lat, lon)

    subject = f"[AlienSight Alert] {person_count} {label} Detected"
    body = (
        f"{person_count} {label} detected by UAV.\n\n"
        f"Location\n"
        f"  Latitude  : {lat:.6f}\n"
        f"  Longitude : {lon:.6f}\n"
        f"  City      : {city}\n"
        f"  Maps      : {maps_link}\n\n"
        f"Timestamp : {timestamp}\n"
    )

    msg            = MIMEMultipart()
    msg["From"]    = sender_email
    msg["To"]      = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if annotated_frame is not None:
        _, buf = cv2.imencode(".jpg", annotated_frame)
        attachment = MIMEImage(buf.tobytes(), name="detection.jpg")
        attachment.add_header("Content-Disposition", "attachment",
                              filename="detection.jpg")
        msg.attach(attachment)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print(f"[alert] Sent to {recipient_email} — {person_count} {label} at {city}")
        return True
    except Exception as e:
        print(f"[alert] Failed to send: {e}")
        return False
