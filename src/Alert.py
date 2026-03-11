import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_alert(lat, lon, confidence, recipient_email, sender_email, sender_password):
    """
    Send an email alert with the GPS location of a detected person.
    Uses Gmail — make sure to use an App Password, not your regular password.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    maps_link = f"https://maps.google.com/?q={lat},{lon}"

    subject = "DRONE ALERT: Person Detected"
    body = f"""
Person Detected!

Location:
  - Latitude:  {lat:.6f}
  - Longitude: {lon:.6f}
  - Google Maps: {maps_link}

Detection confidence: {confidence:.0%}
Time: {timestamp}

---
Sent automatically by the UAV detection system.
    """

    msg = MIMEMultipart()
    msg["From"]    = sender_email
    msg["To"]      = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print(f"Alert sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send alert: {e}")
