# 🚁 UAV Human Detection & Geolocation

A real-time system for detecting humans from drone footage and calculating their GPS coordinates for search-and-rescue operations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

<p align="center">
  <img src="assets/demo.gif" alt="Detection Demo" width="600"/>
</p>

**A real-time system to detect humans from UAV footage and calculate their precise GPS coordinates for emergency response.**

[📄 View Full Technical Report (PDF)](docs/Your_Project_Report.pdf) | [📧 Sample Alert Report](assets/email_alert_screenshot.png)

---

## 🌟 Project Overview
This project was developed for the **CVaPR (Computer Vision and Pattern Recognition)** course. It addresses the critical need for automated victim localization in Search-and-Rescue (SAR) missions. By combining **YOLOv8n** for computer vision and a **custom georeferencing algorithm**, the system can pinpoint a person's location on the globe using only a drone's camera and telemetry data.

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/uav-human-localization.git
cd uav-human-localization
pip install -r requirements.txt

# Run detection
python src/main.py --video path/to/video.mp4 --altitude 50
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Precision | 0.717 |
| Recall | 0.534 |
| mAP@0.5 | 0.585 |

Trained on VisDrone dataset (6,471 images, 50 epochs, 960×960 resolution).

## 📖 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Usage Examples](docs/USAGE.md)
- [Geolocation Algorithm](docs/GEOLOCATION.md)
- [Training Guide](docs/TRAINING.md)
- [API Reference](docs/API.md)

## 🗺️ How It Works

1. **Detect** humans using YOLOv8n
2. **Calculate** GPS coordinates from bounding boxes + drone telemetry
3. **Alert** rescue teams via email with precise location

```
Drone Camera → YOLOv8 → Geolocation Script → GPS Coords → Alert System
```

## 🛠️ Usage

```python
from src.detector import HumanDetector
from src.geolocation import calculate_gps

# Initialize detector
detector = HumanDetector('models/best.pt')

# Process frame
detections = detector.detect(frame)

# Get GPS coordinates
gps_coords = calculate_gps(
    bbox=detections[0],
    drone_pos=(lat, lon, altitude),
    heading=45
)
```

## 🎓 Project Context

Developed as part of CVaPR (Computer Vision and Pattern Recognition) course.  
Tested on DJI Air 3S drone under various real-world conditions.

## 👥 Team

Hamza Ghitri • Wojciech Seman • Krzysztof Połeć • Jakub Gutt • Mohamed Bendimerad
<img width="469" height="316" alt="team" src="https://github.com/user-attachments/assets/40d269b0-e2fa-42fd-8d46-b4cfbc2fb678" />


## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---
