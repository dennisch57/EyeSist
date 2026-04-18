"""
Evaluate the candidate model against production on the held-out test set.
Promote the candidate to production if it is meaningfully better and exceeds
the minimum accuracy floor.

Azure storage layout:
  models/backbone/candidate/ethxgaze_candidate_{timestamp}.pth  ← temporary, always deleted after eval
  models/backbone/promoted/ethxgaze_v{N}_{timestamp}.pth        ← permanent versioned history
  models/backbone/ethxgaze_backbone.pth                         ← production (always latest promoted)
  models/backbone/version.json                                  ← {version, promoted_blob, timestamp}

Test set: original TEST_DIR + cumulative user sessions with split='test' from manifest.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms

from azure_storage import (
    delete_blob,
    download_backbone,
    download_bytes,
    list_session_images,
    load_manifest,
    load_model_version,
    save_last_retrain_info,
    save_model_version,
    upload_backbone,
    upload_bytes,
    upload_json,
)
from model import GazeClassifier, _ResizeWithPad
from runtime_config import get_runtime_device
from training_config import (
    BATCH_SIZE,
    IMG_SIZE,
    LABELS,
    NUM_CLASSES,
    NUM_WORKERS,
    RETRAIN_EXPERIMENT_CONFIG,
    TEST_DIR,
)

PROMOTION_MIN_DELTA = 0.05  # candidate must beat production by at least 5%
PROMOTION_LOG_BLOB = "models/backbone/promotion_log.json"


# ── Dataset helpers ────────────────────────────────────────────────────────────

class _UserTestDataset(Dataset):
    def __init__(self, samples: list[tuple[bytes, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_bytes, label_idx = self.samples[idx]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self.transform(img), label_idx


def _label_from_blob_name(blob_name: str) -> str | None:
    parts = blob_name.split("/")
    if len(parts) < 2:
        return None
    label = parts[-2]
    return label if label in LABELS else None


def _download_user_test_images() -> list[tuple[bytes, int]]:
    """Load test-split session IDs from the manifest and download their images."""
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    manifest = load_manifest()
    test_ids = [e["session_id"] for e in manifest if e.get("split") == "test"]
    print(f"Found {len(test_ids)} test-split sessions in manifest")

    samples: list[tuple[bytes, int]] = []
    for sid in test_ids:
        for blob_path in list_session_images(sid):
            label = _label_from_blob_name(blob_path)
            if label is None:
                continue
            try:
                samples.append((download_bytes(blob_path), label_to_idx[label]))
            except Exception as e:
                print(f"  Warning: skipping {blob_path}: {e}")

    print(f"Downloaded {len(samples)} user test images")
    return samples


def _build_test_loader(user_samples: list[tuple[bytes, int]]) -> DataLoader:
    eval_tf = transforms.Compose([
        _ResizeWithPad(IMG_SIZE, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    original_test = datasets.ImageFolder(TEST_DIR, transform=eval_tf)

    if user_samples:
        user_dataset = _UserTestDataset(user_samples, eval_tf)
        test_dataset = ConcatDataset([original_test, user_dataset])
        print(
            f"Test: {len(original_test)} original + {len(user_dataset)} user "
            f"= {len(test_dataset)} total"
        )
    else:
        test_dataset = original_test
        print(f"Test: {len(original_test)} original only")

    return DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model_from_bytes(model_bytes: bytes, device) -> GazeClassifier:
    checkpoint = torch.load(io.BytesIO(model_bytes), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cfg = checkpoint.get("exp_cfg", RETRAIN_EXPERIMENT_CONFIG)
    from model import resnet50 as _resnet50
    model = GazeClassifier(
        backbone=_resnet50(),
        num_classes=NUM_CLASSES,
        head_dense_units=cfg.get("head_dense_units", []),
        dropout=cfg.get("dropout", 0.1),
        batch_norm_in_head=cfg.get("batch_norm_in_head", True),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model: GazeClassifier, loader: DataLoader, device) -> float:
    correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        correct += model(images).argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


# ── Entry point ────────────────────────────────────────────────────────────────

def evaluate_and_promote(candidate_blob: str, candidate_val_accuracy: float) -> dict:
    """
    Evaluate candidate vs production on the test set and promote if better.

    Args:
        candidate_blob: Azure blob path for the candidate checkpoint.
        candidate_val_accuracy: Val accuracy reported by the training step.

    Returns:
        {promoted, candidate_test_acc, production_test_acc, timestamp}
    """
    device = get_runtime_device()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    from clearml import Task
    task = Task.current_task()
    if task is None:
        task = Task.init(
            project_name="EyeSist",
            task_name=f"evaluate_promote_{timestamp}",
            task_type=Task.TaskTypes.qc,
            reuse_last_task_id=False,
        )
    logger = task.get_logger()

    # Build test loader
    user_test = _download_user_test_images()
    test_loader = _build_test_loader(user_test)

    # Load both models
    print("Loading candidate model from Azure")
    candidate_bytes = download_bytes(candidate_blob)
    candidate_model = _load_model_from_bytes(candidate_bytes, device)

    print("Loading production model from Azure")
    production_bytes = download_backbone()
    production_model = _load_model_from_bytes(production_bytes, device)

    # Evaluate
    candidate_test_acc = _evaluate(candidate_model, test_loader, device)
    production_test_acc = _evaluate(production_model, test_loader, device)

    print(f"Candidate  test acc : {candidate_test_acc:.4f}  (val: {candidate_val_accuracy:.4f})")
    print(f"Production test acc : {production_test_acc:.4f}")

    logger.report_single_value("candidate_val_accuracy", candidate_val_accuracy)
    logger.report_single_value("candidate_test_accuracy", candidate_test_acc)
    logger.report_single_value("production_test_accuracy", production_test_acc)

    # Promotion decision — promote if candidate beats production by at least MIN_DELTA
    promoted = candidate_test_acc >= production_test_acc + PROMOTION_MIN_DELTA

    if promoted:
        # Increment version and write to permanent versioned path
        version_info = load_model_version()
        next_version = version_info["version"] + 1
        promoted_blob = f"models/backbone/promoted/ethxgaze_v{next_version}_{timestamp}.pth"

        print(
            f"Promoting candidate → v{next_version} "
            f"({candidate_test_acc:.4f} > production test acc: {production_test_acc:.4f} + min delta: {PROMOTION_MIN_DELTA})"
        )

        upload_bytes(promoted_blob, candidate_bytes)          # versioned permanent copy
        upload_backbone(candidate_bytes)                      # overwrite production blob
        save_model_version({
            "version": next_version,
            "promoted_blob": promoted_blob,
            "timestamp": timestamp,
            "candidate_test_acc": candidate_test_acc,
            "production_test_acc": production_test_acc,
        })
        print(f"Production backbone updated → {promoted_blob}")

        # Record retrain baseline so check_retrain counts only NEW sessions next run
        manifest = load_manifest()
        train_entries = [e for e in manifest if e.get("split") == "train"]
        save_last_retrain_info({
            "timestamp": timestamp,
            "num_train_sessions": len(train_entries),
            "num_train_samples": sum(e.get("num_samples", 0) for e in train_entries),
        })

        # Clear in-process singleton so FastAPI reloads on next request
        try:
            import model as _model_module
            _model_module._model = None
        except Exception:
            pass
    else:
        print(
            f"Candidate NOT promoted — not enough improvement "
            f"({candidate_test_acc:.4f} vs {production_test_acc:.4f}, "
            f"need +{PROMOTION_MIN_DELTA})."
        )

    # Delete candidate blob regardless of outcome — keeps storage clean
    try:
        delete_blob(candidate_blob)
        print(f"Deleted candidate blob: {candidate_blob}")
    except Exception as e:
        print(f"Warning: could not delete candidate blob: {e}")

    # Append to promotion log
    promotion_record = {
        "timestamp": timestamp,
        "promoted": promoted,
        "candidate_val_accuracy": candidate_val_accuracy,
        "candidate_test_accuracy": candidate_test_acc,
        "production_test_accuracy": production_test_acc,
        "promoted_blob": promoted_blob if promoted else None,
    }
    try:
        existing = []
        try:
            from azure_storage import download_json
            existing = download_json(PROMOTION_LOG_BLOB)
        except Exception:
            pass
        existing.append(promotion_record)
        upload_json(PROMOTION_LOG_BLOB, existing)
    except Exception as e:
        print(f"Warning: could not update promotion log: {e}")

    logger.report_single_value("promoted", int(promoted))
    task.upload_artifact("promotion_record", promotion_record)

    return {
        "promoted": promoted,
        "candidate_test_acc": candidate_test_acc,
        "production_test_acc": production_test_acc,
        "timestamp": timestamp,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-blob", required=True)
    parser.add_argument("--val-accuracy", type=float, required=True)
    args = parser.parse_args()
    print(evaluate_and_promote(args.candidate_blob, args.val_accuracy))
