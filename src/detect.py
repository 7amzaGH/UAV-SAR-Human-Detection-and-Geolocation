"""
detect.py
---------
YOLOv8n inference wrapper for human detection.
"""

import cv2
import numpy as np

class PTDetector:
    """
    PyTorch / Ultralytics Implementation.
    Optimized for NVIDIA Jetson (GPU) or Desktop environments.
    """
    def __init__(self, model_path, conf_threshold=0.6):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
        except ImportError:
            raise ImportError("Please install 'ultralytics' to use PTDetector.")

    def detect(self, frame):
        # verbose=False keeps the console clean during the mission
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = []
        for r in results.boxes:
            # Get coordinates and convert to integers
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = float(r.conf[0])
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "conf": round(conf, 4)
            })
        return detections


class ONNXDetector:
    """
    ONNX Runtime Implementation.
    Optimized for NPU (Qualcomm/Edge) or CPU-only environments.
    """
    def __init__(self, model_path, provider="CPUExecutionProvider", conf_threshold=0.6):
        try:
            import onnxruntime as ort
            # We try the requested provider (e.g., QNN), fall back to CPU if hardware is missing
            available = ort.get_available_providers()
            selected = provider if provider in available else "CPUExecutionProvider"
            
            self.session = ort.InferenceSession(model_path, providers=[selected])
            self.input_name = self.session.get_inputs()[0].name
            self.conf_threshold = conf_threshold
        except ImportError:
            raise ImportError("Please install 'onnxruntime' to use ONNXDetector.")

    def _preprocess(self, frame):
        """Prepares the frame for ONNX inference (960x960 square)."""
        # Resize to standard YOLO square input
        img = cv2.resize(frame, (960, 960))
        # BGR to RGB and normalize to [0, 1]
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        # HWC to CHW format
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension [1, 3, 960, 960]
        return np.expand_dims(img, axis=0)

    def detect(self, frame):
        blob = self._preprocess(frame)
        output = self.session.run(None, {self.input_name: blob})[0]

        # scale factor to map 960x960 back to original frame size
        scale_x = frame.shape[1] / 960
        scale_y = frame.shape[0] / 960

        detections = []
        # Expected output shape: [1, num_boxes, 6] (x1, y1, x2, y2, conf, cls)
        for det in output[0]:
            conf = float(det[4])
            if conf >= self.conf_threshold:
                x1 = int(det[0] * scale_x)
                y1 = int(det[1] * scale_y)
                x2 = int(det[2] * scale_x)
                y2 = int(det[3] * scale_y)
                
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "conf": round(conf, 4)
                })
        return detections


def draw_detections(frame, detections, color=(255, 100, 0), thickness=2):
    """
    Unified drawing utility for both detectors.
    """
    annotated_frame = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        
        # Bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label
        label = f"human {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
    return annotated_frame