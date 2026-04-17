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


def list_all_session_metas() -> list[dict]:
    """Return all session meta.json contents."""
    metas = []
    for blob_path in list_blobs("calibration-data/"):
        if blob_path.endswith("/meta.json"):
            try:
                metas.append(download_json(blob_path))
            except Exception:
                pass
    return metas
