from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from collections import Counter, deque
import base64
import cv2
import numpy as np
import pickle
import os
import json

from model import extract_features, predict_base
from eye_detector import detect_eyes
from pipeline.step_calibrate_user import calibrate_user, SESSION_CACHE_DIR
from runtime_config import get_runtime_device_str
from training_config import LABELS

app = FastAPI()
SMOOTHING_WINDOW = 10  # Number of recent frames to use for majority-vote smoothing (tunable)


@app.on_event("startup")
def log_runtime_device() -> None:
    print(f"EyeSist backend starting on device: {get_runtime_device_str()}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: specify allowed origins (good for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None


class Frame(BaseModel):
    image: str  # base64 data URL


class CalibrateRequest(BaseModel):
    """Calibration request with image crops for each gaze direction."""
    crops: dict[str, list[str]]  # {label: [base64_images...], ...}


class PredictRequest(BaseModel):
    """Prediction request with optional session_id for user-calibrated Ridge model."""
    image: str  # base64
    session_id: str | None = None


def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode raw encoded image bytes to BGR ndarray."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image bytes")
    return img_bgr


def base64_to_bgr(b64_str: str) -> np.ndarray:
    """Decode base64 data URL or raw base64 to BGR ndarray."""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_data = base64.b64decode(b64_str)
        return bytes_to_bgr(img_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")


def dummy_model(img: np.ndarray) -> str:
    del img
    return np.random.choice(["left", "right", "up", "down", "closed", "open"]).item()


def predict_from_eye_crops(eye_crops: list[np.ndarray], session_id: str | None) -> list[str]:
    if session_id:
        model_path = os.path.join(SESSION_CACHE_DIR, f"{session_id}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Session {session_id} not found or expired")

        with open(model_path, "rb") as f:
            ridge_clf = pickle.load(f)

        features = [extract_features(crop) for crop in eye_crops]
        pred_indices = ridge_clf.predict(features)
        return [LABELS[pred_idx] for pred_idx in pred_indices]

    return [predict_base(crop) for crop in eye_crops]


def infer_frame(image_b64: str, session_id: str | None = None) -> dict:
    img_bgr = base64_to_bgr(image_b64)
    return infer_frame_bgr(img_bgr, session_id)


def infer_frame_bgr(img_bgr: np.ndarray, session_id: str | None = None) -> dict:
    """Run eye detection and gaze inference on a decoded frame."""

    detections = detect_eyes(img_bgr)
    if not detections:
        return {
            "error": "No eye detected",
            "gaze": None,
            "boxes": [],
            "frame_size": {"width": int(img_bgr.shape[1]), "height": int(img_bgr.shape[0])},
        }

    eye_crops = [detection["crop"] for detection in detections]
    boxes = [detection["box"] for detection in detections]

    try:
        gaze = predict_from_eye_crops(eye_crops, session_id)
    except FileNotFoundError as exc:
        return {
            "error": str(exc),
            "gaze": None,
            "boxes": boxes,
            "frame_size": {"width": int(img_bgr.shape[1]), "height": int(img_bgr.shape[0])},
        }

    return {
        "gaze": gaze,
        "boxes": boxes,
        "frame_size": {"width": int(img_bgr.shape[1]), "height": int(img_bgr.shape[0])},
    }


def smooth_gaze(history: deque[list[str]], gaze: list[str]) -> tuple[list[str], list[float]]:
    """Apply majority-vote smoothing over the last few gaze predictions."""
    history.append(gaze)

    smoothed: list[str] = []
    confidences: list[float] = []
    for eye_idx in range(len(gaze)):
        labels = [frame_gaze[eye_idx] for frame_gaze in history if len(frame_gaze) > eye_idx]
        label, count = Counter(labels).most_common(1)[0]
        smoothed.append(label)
        confidences.append(count / len(labels))

    return smoothed, confidences


def collapse_gaze(gaze: list[str], confidences: list[float]) -> str:
    """Return one final label from the per-eye smoothed predictions."""
    if not gaze:
        raise ValueError("Expected at least one gaze label to collapse.")

    best_idx = max(range(len(gaze)), key=lambda idx: confidences[idx] if idx < len(confidences) else 0.0)
    return gaze[best_idx]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.post("/calibrate")
async def calibrate(
    files: List[UploadFile] = File(...),
    labels: List[str] = Form(...)
):
    data = {}

    for file, label in zip(files, labels):
        contents = await file.read()

        if label not in data:
            data[label] = []

        data[label].append(contents)

    print({k: len(v) for k, v in data.items()})

    return {"status": "received"}


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    session_id: str | None = None
    gaze_history: deque[list[str]] = deque(maxlen=SMOOTHING_WINDOW)

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                frame_bytes = message["bytes"]
                img_bgr = bytes_to_bgr(frame_bytes)
                result = infer_frame_bgr(img_bgr, session_id)
                if result.get("gaze"):
                    smoothed_gaze, confidences = smooth_gaze(gaze_history, result["gaze"])
                    result["gaze"] = collapse_gaze(smoothed_gaze, confidences)
                else:
                    gaze_history.clear()
                await websocket.send_json(result)
                continue

            if "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])
                session_id = payload.get("session_id", session_id)
                image = payload.get("image")

                if image:
                    result = infer_frame(image, session_id)
                    if result.get("gaze"):
                        smoothed_gaze, confidences = smooth_gaze(gaze_history, result["gaze"])
                        result["gaze"] = collapse_gaze(smoothed_gaze, confidences)
                    else:
                        gaze_history.clear()
                    await websocket.send_json(result)
                else:
                    ready_payload = {"status": "ready", "session_id": session_id}
                    await websocket.send_json(ready_payload)
                continue

            error_payload = {"error": "Unsupported WebSocket payload", "gaze": None}
            await websocket.send_json(error_payload)

    except WebSocketDisconnect:
        print("WebSocket /ws/predict closed")
    except json.JSONDecodeError:
        await websocket.send_json({"error": "Invalid JSON payload", "gaze": None})
    except Exception as e:
        print("WebSocket /ws/predict error:", e)

@app.post("/predict")
def predict(frame: Frame):
    img_bgr = base64_to_bgr(frame.image)
    direction = dummy_model(img_bgr)
    return {"direction": direction}


@app.post("/v2/calibrate")
def calibrate_v2(req: CalibrateRequest):
    """
    Calibrate a user session from crops for each gaze direction.
    Returns session_id and pre-calibration accuracy.
    Ridge model stored locally (TODO: move to frontend IndexedDB).
    """
    try:
        calibration_crops = {}
        for label, b64_list in req.crops.items():
            crops = [base64_to_bgr(b64) for b64 in b64_list]
            calibration_crops[label] = crops

        session_id, result = calibrate_user(calibration_crops)

        return {
            "session_id": session_id,
            "accuracy": result["accuracy"],
            "num_samples": result["num_samples"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
