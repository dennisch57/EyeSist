import os
import json
import pickle
import io
from functools import lru_cache

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "anonifyme")
ACCOUNT_URL =  os.getenv("AZURE_BLOB_URL", f"https://{ACCOUNT_NAME}.blob.core.windows.net")
CONTAINER = "eyesist"

# Blob path constants
BACKBONE_BLOB = "models/backbone/ethxgaze_backbone.pth"
YOLO_BLOB = "models/yolo/yolo_eye.pt"
MANIFEST_BLOB = "dataset/manifest.json"
LAST_RETRAIN_BLOB = "dataset/last_retrain.json"
MODEL_VERSION_BLOB = "models/backbone/version.json"


@lru_cache(maxsize=1)
def _credential():
    return DefaultAzureCredential()


@lru_cache(maxsize=1)
def _client() -> BlobServiceClient:
    return BlobServiceClient(account_url=ACCOUNT_URL, credential=_credential())


def ensure_container() -> None:
    container = _client().get_container_client(CONTAINER)
    if not container.exists():
        container.create_container()


def upload_bytes(blob_path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    blob = _client().get_blob_client(container=CONTAINER, blob=blob_path)
    blob.upload_blob(data, overwrite=True,
                     content_settings=ContentSettings(content_type=content_type))


def upload_json(blob_path: str, data: dict) -> None:
    upload_bytes(blob_path, json.dumps(data).encode(), "application/json")


def upload_file(blob_path: str, local_path: str) -> None:
    with open(local_path, "rb") as f:
        upload_bytes(blob_path, f.read())


def download_bytes(blob_path: str) -> bytes:
    blob = _client().get_blob_client(container=CONTAINER, blob=blob_path)
    return blob.download_blob().readall()


def download_json(blob_path: str) -> dict:
    return json.loads(download_bytes(blob_path))


def download_file(blob_path: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(download_bytes(blob_path))


def blob_exists(blob_path: str) -> bool:
    blob = _client().get_blob_client(container=CONTAINER, blob=blob_path)
    return blob.exists()


def list_blobs(prefix: str) -> list[str]:
    container = _client().get_container_client(CONTAINER)
    return [b.name for b in container.list_blobs(name_starts_with=prefix)]


def list_session_images(session_id: str) -> list[str]:
    return list_blobs(f"calibration-data/{session_id}/images/")


def upload_calibration_images(session_id: str, crops: list[tuple[str, bytes]]) -> None:
    """Upload list of (label, jpeg_bytes) crops for a session."""
    label_counts: dict[str, int] = {}
    for label, jpeg_bytes in crops:
        idx = label_counts.get(label, 0)
        label_counts[label] = idx + 1
        blob_path = f"calibration-data/{session_id}/images/{label}_{idx:04d}.jpg"
        upload_bytes(blob_path, jpeg_bytes, "image/jpeg")


def upload_session_meta(session_id: str, meta: dict) -> None:
    upload_json(f"calibration-data/{session_id}/meta.json", meta)


def download_session_meta(session_id: str) -> dict:
    return download_json(f"calibration-data/{session_id}/meta.json")


def load_manifest() -> list[dict]:
    """Return the full dataset manifest, or [] if it doesn't exist yet."""
    try:
        return download_json(MANIFEST_BLOB)
    except Exception:
        return []


def append_to_manifest(entry: dict) -> None:
    """Append one session entry to the manifest. Thread-unsafe — fine for sequential ingestion."""
    manifest = load_manifest()
    manifest.append(entry)
    upload_json(MANIFEST_BLOB, manifest)


def load_last_retrain_info() -> dict:
    """Return info about the last completed retrain, or defaults if none yet."""
    try:
        return download_json(LAST_RETRAIN_BLOB)
    except Exception:
        return {"timestamp": None, "num_train_sessions": 0, "num_train_samples": 0}


def save_last_retrain_info(info: dict) -> None:
    upload_json(LAST_RETRAIN_BLOB, info)


def update_session_meta(session_id: str, updates: dict) -> None:
    """Patch an existing session meta.json with the given fields."""
    meta = download_session_meta(session_id)
    meta.update(updates)
    upload_session_meta(session_id, meta)


def upload_backbone(data: bytes) -> None:
    """Overwrite the production backbone checkpoint on Azure."""
    upload_bytes(BACKBONE_BLOB, data)


def download_backbone() -> bytes:
    return download_bytes(BACKBONE_BLOB)


def delete_blob(blob_path: str) -> None:
    blob = _client().get_blob_client(container=CONTAINER, blob=blob_path)
    blob.delete_blob()


def load_model_version() -> dict:
    """Return current model version info, or defaults if no version exists yet."""
    try:
        return download_json(MODEL_VERSION_BLOB)
    except Exception:
        return {"version": 0, "promoted_blob": None, "timestamp": None}


def save_model_version(info: dict) -> None:
    upload_json(MODEL_VERSION_BLOB, info)


def list_all_session_metas() -> list[dict]:
    """Return all session meta.json contents, with session_id injected from blob path."""
    metas = []
    for blob_path in list_blobs("calibration-data/"):
        if blob_path.endswith("/meta.json"):
            try:
                meta = download_json(blob_path)
                if "session_id" not in meta:
                    meta["session_id"] = blob_path.split("/")[1]
                metas.append(meta)
            except Exception:
                pass
    return metas
