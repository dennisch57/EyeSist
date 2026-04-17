import os
import numpy as np
import cv2
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "yolo26_eye_detector.pt")
DETECTION_MAX_DIM = 960
DETECTION_DEVICE = "cpu" # too heavy in memory for GPU inference, and CPU is fast enough for our small model

_yolo: YOLO | None = None


def get_yolo() -> YOLO:
    global _yolo
    if _yolo is None:
        _yolo = YOLO(MODEL_PATH)
        _yolo.to(DETECTION_DEVICE)
    return _yolo


def _resize_for_detection(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = img_bgr.shape[:2]
    max_dim = max(height, width)
    if max_dim <= DETECTION_MAX_DIM:
        return img_bgr, 1.0

    scale = DETECTION_MAX_DIM / max_dim
    resized = cv2.resize(
        img_bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def detect_eyes(img_bgr: np.ndarray) -> list[dict[str, object]]:
    """
    Detect up to two eyes and return left-to-right eye detections with
    original-frame bounding boxes and crops.
    """
    detection_img, scale = _resize_for_detection(img_bgr)
    results = get_yolo()(detection_img, verbose=False, device=DETECTION_DEVICE)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    top_indices = boxes.conf.argsort(descending=True)[:2].tolist()
    selected_boxes = sorted(
        (boxes[i] for i in top_indices),
        key=lambda box: float(box.xyxy[0][0]),
    )

    detections: list[dict[str, object]] = []
    for box in selected_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if scale != 1.0:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

        x1 = max(0, min(x1, img_bgr.shape[1]))
        x2 = max(0, min(x2, img_bgr.shape[1]))
        y1 = max(0, min(y1, img_bgr.shape[0]))
        y2 = max(0, min(y2, img_bgr.shape[0]))
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size != 0:
            detections.append(
                {
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "crop": crop,
                }
            )

    return detections


def crop_eye(img_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Detect up to two eyes in the frame and return valid crops ordered left-to-right.
    Returns an empty list if no eyes are detected.
    """
    return [detection["crop"] for detection in detect_eyes(img_bgr)]
