import os
import numpy as np
import cv2
from ultralytics import YOLO

from runtime_config import get_runtime_device_str

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "yolo26_eye_detector.pt")

_yolo: YOLO | None = None


def get_yolo() -> YOLO:
    global _yolo
    if _yolo is None:
        _yolo = YOLO(MODEL_PATH)
        _yolo.to(get_runtime_device_str())
    return _yolo


def crop_eye(img_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Detect up to two eyes in the frame and return valid crops ordered left-to-right.
    Returns an empty list if no eyes are detected.
    """
    results = get_yolo()(img_bgr, verbose=False, device=get_runtime_device_str())
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    top_indices = boxes.conf.argsort(descending=True)[:2].tolist()
    selected_boxes = sorted(
        (boxes[i] for i in top_indices),
        key=lambda box: float(box.xyxy[0][0]),
    )

    crops: list[np.ndarray] = []
    for box in selected_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size != 0:
            crops.append(crop)

    return crops
