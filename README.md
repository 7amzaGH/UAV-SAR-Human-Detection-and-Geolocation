# UAV Human Detection & Geolocation

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

## Project Overview
This project was developed for the **CVaPR (Computer Vision and Pattern Recognition)** course. It addresses the critical need for automated missing persons localization in Search-and-Rescue (SAR) missions. By combining **YOLOv8n** for computer vision and a **custom georeferencing algorithm**, the system can pinpoint a person's location on the globe using only a drone's camera and telemetry data.

<p align="center">
  <img src="assets/demo.gif" width="800" alt="UAV Human Detection Demo">
  <br>
  <em>Real-time detection and coordinate estimation using DJI Air 3S footage.</em>
</p>

![DJI-Air-3-drone-800x407](https://github.com/user-attachments/assets/38274368-43da-459b-9527-3a11184d3152)

---

## How It Works

### 1. Detection (YOLOv8n)
We trained a **YOLOv8 Nano** model on a filtered subset of the **VisDrone dataset**. 
* **Classes:** Merged "pedestrian" and "people" into a single `human` class.
* **Input Resolution:** 960x960 pixels.
* **Platform:** Optimized for edge deployment (e.g., NVIDIA Jetson Nano).

### 2. Geolocation Algorithm
The system translates 2D image coordinates into 3D GPS coordinates by calculating the ray-intersection between the camera lens and the ground plane.



**Key variables utilized:**
* **UAV Telemetry:** Current Latitude, Longitude, and Altitude (m).
* **Attitude:** Drone Heading ($\psi$) and Gimbal Pitch ($\theta$).
* **Camera Geometry:** Field of View (FOV) and pixel-to-meter scaling.

### 3. Automated Alerting
Once a human is detected with high confidence, the `src/alert.py` module triggers an automated email to rescue teams containing:
* Precise **GPS Link** (Google Maps format).
* **Timestamp** and Drone Telemetry.
* **Visual Snapshot** of the detection for verification.

---



## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/uav-human-localization.git
cd uav-human-localization
pip install -r requirements.txt

# Run detection
python src/main.py --video path/to/video.mp4 --altitude 50
```

## Model Training & Performance

The core of this system is a custom-trained **YOLOv8n** (Nano) model. This architecture was selected to ensure high-speed inference on edge computing hardware while maintaining the precision required for life-critical search-and-rescue operations.

### Training Phase
The model was trained using a refined version of the **VisDrone Dataset** to optimize it for aerial perspectives.

* **Dataset Source:** [VisDrone (Kaggle)](https://www.kaggle.com/datasets/vigneshp6/visdrone-dataset)
* **Dataset Volume:** 6,471 images.
* **Preprocessing:** The "pedestrian" and "people" classes were merged into a single **"human"** class to increase detection density and simplify the output for rescue teams.
* **Training Parameters:**
    * **Epochs:** 50
    * **Input Resolution:** 960 × 960 pixels.
    * **Hardware:** Trained on NVIDIA L4 GPU (Google Colab).



---

### 🧪 Real-World Evaluation
To bridge the gap between dataset training and practical application, we conducted field tests using the **DJI Air 3S Fly More Combo (DJI RC-N3)**. 

We recorded **4 unique test videos** in outdoor environments, specifically varying the **altitudes (15m–30m)** and **gimbal angles (45°–90°)** to test the model's robustness against perspective distortion.
<img width="1352" height="316" alt="image" src="https://github.com/user-attachments/assets/dad66574-924c-4442-b9a3-9cd87d232a65" />


#### Performance Metrics (at Threshold = 0.4)
On our self-collected real-world dataset, the model achieved the following results:

| Metric | Value |
| :--- | :--- |
| **Precision** | **0.987** |
| **Recall** | **0.904** |
| **mAP @ 0.5** | **0.887** |
| **mAP @ 0.5:0.95** | **0.717** |


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

## 👥 Team

Hamza Ghitri • Wojciech Seman • Krzysztof Połeć • Jakub Gutt • Mohamed Bendimerad
<img width="469" height="316" alt="team" src="https://github.com/user-attachments/assets/40d269b0-e2fa-42fd-8d46-b4cfbc2fb678" />


## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---
