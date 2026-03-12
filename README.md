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

**The full pipeline:**

```
Drone Video Feed  +  GPS  +  Altitude  +  Heading
          │
          ▼
┌─────────────────────────┐
│    YOLOv8n Detection    │  ← Fine-tuned on VisDrone
│  Output: Bounding Box   │    "pedestrian" + "people" → "human"
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Geolocation Module    │  ← Pixel offset → meters (FOV + altitude)
│  Output: (lat, lon)     │    Heading rotation → GPS delta conversion
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Email Alert System   │  ← GPS coords + timestamp + map link
│  Output: Alert to team  │    Sent instantly via Gmail SMTP
└─────────────────────────┘
```


<p align="center">
  <img src="https://github.com/user-attachments/assets/38274368-43da-459b-9527-3a11184d3152" width="800" alt="UAV Human Detection Demo">
  <br>
  <em>Real-time detection and coordinate estimation using DJI Air 3S footage.</em>
</p>



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

## Geolocation Algorithm

The core contribution of this project. The system translates a 2D bounding box pixel coordinate into a real-world GPS coordinate through three steps:

**Step 1 — Compute image scale (meters/pixel)**

Using the drone's altitude and camera field of view:
```
scale_x = (altitude × tan(FOV_H / 2)) / (IMAGE_WIDTH  / 2)
scale_y = (altitude × tan(FOV_V / 2)) / (IMAGE_HEIGHT / 2)
```

**Step 2 — Rotate offset by drone heading**

The pixel offset from the image center is rotated to align with true north using compass heading θ:
```
x' = cos(θ) × dx  −  sin(θ) × dy
y' = sin(θ) × dx  +  cos(θ) × dy
```

**Step 3 — Convert meters → GPS coordinates**

Using Earth's circumference (40,075 km) with a cosine correction for longitude:
```
Δlat = (y_meters / 40,075,000) × 360
Δlon = (x_meters / (40,075,000 × cos(lat_rad))) × 360

final_lat = drone_lat + Δlat
final_lon = drone_lon + Δlon
```
<img width="1408" height="768" alt="Geolocation Algorithm" src="https://github.com/user-attachments/assets/db23cd62-3c77-439c-8532-381890afb774" />


**Key inputs required:**

| Input | Description |
|---|---|
| Bounding box center (x, y) | Pixel coordinates of detected human |
| Drone latitude / longitude | From onboard GPS |
| Altitude (meters) | Above ground level |
| Heading (degrees) | 0° = North, clockwise |
| Camera FOV | Horizontal and vertical angles |

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

### Real-World Evaluation
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


## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/uav-human-localization.git
cd uav-human-localization
pip install -r requirements.txt
```

### 2. Run on a recorded video (local testing)
Make sure `Video.MP4` and `Video.SRT` are in the same folder.
The SRT file contains all flight telemetry : GPS, altitude — per frame automatically.
```bash
python src/main_local.py
```

### 3. Run live on the drone (Jetson Nano)
GPS, altitude and heading are read in real-time from the flight controller via serial.
```bash
python src/main_live.py
```

---

## Usage in Python

```python
from src.detect import HumanDetector
from src.geolocation import get_real_coords

# Initialize detector
detector = HumanDetector('models/best.pt', conf_threshold=0.4)

# Run detection on a frame
detections = detector.detect(frame)

# Get GPS coordinates for first detection
if detections:
    lat, lon = get_real_coords(
        bbox_center    = detections[0]["center"],
        drone_position = (50.2648, 19.0237, 30),  # (lat, lon, altitude_m)
        drone_heading  = 45,
        camera_config  = {"fov_h": 84, "fov_v": 54,
                          "image_width": 1920, "image_height": 1080}
    )
    print(f"Human detected at: {lat:.6f}, {lon:.6f}")
```

## Team

Hamza Ghitri • Wojciech Seman • Krzysztof Połeć • Jakub Gutt • Mohamed Bendimerad
<img width="469" height="316" alt="team" src="https://github.com/user-attachments/assets/40d269b0-e2fa-42fd-8d46-b4cfbc2fb678" />


## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@misc{uav_sar_2025,
  title   = {UAV Human Detection and Geolocation for Search-and-Rescue Operations},
  author  = {Ghitri, Hamza and Seman, Wojciech and Połeć, Krzysztof and Gutt, Jakub and Bendimerad, Mohamed},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/uav-human-localization},
  note    = {CVaPR Course Project}
}
```

---

## Acknowledgments

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) — Tianjin University
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built for search-and-rescue applications. Every second counts. 🔴</sub>
</div>

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---
