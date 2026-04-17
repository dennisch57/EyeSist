import os
import json
import uuid
import pickle
import cv2
from datetime import datetime
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from model import extract_features
from azure_storage import upload_calibration_images, upload_session_meta

ALPHA = 1.0  # Fixed Ridge alpha (WebGazer.js design)
SESSION_CACHE_DIR = "/tmp/eyesist_sessions"  # TODO: move to IndexedDB on frontend


def calibrate_user(
    calibration_crops: dict[str, list[np.ndarray]],
    session_id: str | None = None,
) -> tuple[str, dict]:
    """
    Fit Ridge Regression on calibration crops for a new user session.
    Stores ridge model locally (TODO: frontend IndexedDB).
    Uploads images + meta to Azure for retraining.

    Args:
        calibration_crops: {label: [crop_image_1, crop_image_2, ...], ...}
        session_id: optional override; if None, generates UUID

    Returns:
        (session_id, ridge_model_dict)
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    os.makedirs(SESSION_CACHE_DIR, exist_ok=True)

    # Extract features for all crops
    X, y = [], []
    crops_to_upload = []

    for label_idx, label in enumerate(['closed', 'down', 'left', 'right', 'straight', 'up']):
        if label not in calibration_crops:
            continue
        crops = calibration_crops[label]
        for crop_bgr in crops:
            try:
                feat = extract_features(crop_bgr)
                X.append(feat)
                y.append(label_idx)
                # Prepare for upload (JPEG)
                _, jpeg = cv2.imencode('.jpg', crop_bgr)
                crops_to_upload.append((label, jpeg.tobytes()))
            except Exception as e:
                print(f"Warning: failed to extract features for {label}: {e}")

    X = np.array(X)
    y = np.array(y)

    if len(X) < 10:
        raise ValueError(f"Need at least 10 calibration samples, got {len(X)}")

    # Fit Ridge
    ridge_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeClassifier(alpha=ALPHA, class_weight="balanced")),
    ])
    ridge_clf.fit(X, y)

    # Calculate pre-calibration accuracy (on same data)
    accuracy = ridge_clf.score(X, y)

    # Store Ridge model locally (TODO: send to frontend IndexedDB)
    local_path = os.path.join(SESSION_CACHE_DIR, f"{session_id}.pkl")
    with open(local_path, "wb") as f:
        pickle.dump(ridge_clf, f)

    # Upload images + meta to Azure (for retraining data)
    upload_calibration_images(session_id, crops_to_upload)
    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_samples": len(X),
        "accuracy": float(accuracy),
        "split": None,  # Will be assigned during retrain check
    }
    upload_session_meta(session_id, meta)

    print(f"✓ Calibrated session {session_id}: {accuracy:.3f} pre-calib accuracy, {len(X)} samples")

    return session_id, {
        "session_id": session_id,
        "accuracy": float(accuracy),
        "num_samples": len(X),
    }
