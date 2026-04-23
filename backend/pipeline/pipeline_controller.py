"""
EyeSist base model retrain pipeline.

Shared training/eval helpers live in backend/pipeline_helpers.py and are
imported inside each component (each component runs as an isolated subprocess;
module-level definitions are NOT available unless imported explicitly).

Pipeline steps:
  1. check_retrain      — volume-based gate
  2. ingest_data        — download user session images to /tmp/
  3. train_resnet50     — train resnet50_layer3 config      ─┐
  4. train_mobilenet    — train mobilenet_v3_large config    ├─ run in parallel
  5. train_efficientnet — train efficientnet_b0 config      ─┘
  6. eval_model         — pick best candidate by val accuracy
  7. get_test_result    — evaluate winner on test set
  8. evaluate_promotion — compare winner vs production on test set
  9. upload_promoted    — upload to Azure if promoted (versioned)
  finally               — clean up /tmp/eyesist_pipeline/

Usage:
  Run locally (cron job):
    python pipeline_controller.py --run-local [--force]

  Run on ClearML agents (NOTE: local /tmp/ paths are not shared across agents):
    python pipeline_controller.py --run-now [--force]
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from clearml import Task
from clearml.automation import PipelineDecorator
from training_config import PIPELINE_TMP_DIR

CLEARML_PROJECT = "EyeSist"
PIPELINE_NAME = "Base Model Retrain"
CONTROLLER_QUEUE = "services"
EXECUTION_QUEUE = "default"
PROMOTION_MIN_DELTA = 0.02


# ── Step 1: check_retrain ──────────────────────────────────────────────────────

@PipelineDecorator.component(
    name="check_retrain",
    return_values=["should_retrain", "train_ids", "val_ids", "test_ids"],
    task_type=Task.TaskTypes.data_processing,
    execution_queue=EXECUTION_QUEUE,
)
def component_check_retrain(force: bool = False) -> tuple:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    from azure_storage import load_last_retrain_info, load_manifest
    from training_config import RETRAIN_NEW_SAMPLES_THRESHOLD

    manifest = load_manifest()
    last = load_last_retrain_info()
    train_entries = [e for e in manifest if e.get("split") == "train"]
    val_entries   = [e for e in manifest if e.get("split") == "val"]
    test_entries  = [e for e in manifest if e.get("split") == "test"]
    current_samples = sum(e.get("num_samples", 0) for e in train_entries)
    prev_samples    = last.get("num_train_samples", 0)
    new_samples     = current_samples - prev_samples
    print(f"Train pool: {current_samples} | since last retrain: +{new_samples} | threshold: {RETRAIN_NEW_SAMPLES_THRESHOLD}")
    if force:
        print("Force flag set — bypassing volume threshold.")
    should_retrain = force or (new_samples >= RETRAIN_NEW_SAMPLES_THRESHOLD)
    return (
        should_retrain,
        [e["session_id"] for e in train_entries],
        [e["session_id"] for e in val_entries],
        [e["session_id"] for e in test_entries],
    )


# ── Step 2: ingest_data ────────────────────────────────────────────────────────

@PipelineDecorator.component(
    name="ingest_data",
    return_values=["user_data_dir"],
    task_type=Task.TaskTypes.data_processing,
    parents=["check_retrain"],
    execution_queue=EXECUTION_QUEUE,
)
def component_ingest_data(train_ids: list, val_ids: list, test_ids: list) -> str:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    from azure_storage import download_bytes, list_session_images
    from training_config import LABELS, PIPELINE_TMP_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR
    from clearml import Task as ClearMLTask

    def _count_images(directory: str) -> int:
        total = 0
        if not os.path.isdir(directory):
            return 0
        for _, _, files in os.walk(directory):
            total += sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
        return total

    orig_counts = {
        "train": _count_images(TRAIN_DIR),
        "val":   _count_images(VAL_DIR),
        "test":  _count_images(TEST_DIR),
    }
    print("Original dataset:")
    for split, n in orig_counts.items():
        print(f"  {split}: {n} images")

    task = ClearMLTask.current_task()
    logger = task.get_logger() if task else None
    if logger:
        for split, n in orig_counts.items():
            logger.report_single_value(f"original_dataset/{split}", n)

    user_data_dir = os.path.join(PIPELINE_TMP_DIR, "user_data")
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    print("User session data:")
    for split, session_ids in split_ids.items():
        split_dir = os.path.join(user_data_dir, split)
        for label in LABELS:
            os.makedirs(os.path.join(split_dir, label), exist_ok=True)
        count = 0
        for sid in session_ids:
            for blob_path in list_session_images(sid):
                parts = blob_path.split("/")
                if len(parts) < 2:
                    continue
                label = parts[-2]
                if label not in LABELS:
                    continue
                local_path = os.path.join(split_dir, label, f"{sid[:8]}_{parts[-1]}")
                try:
                    with open(local_path, "wb") as f:
                        f.write(download_bytes(blob_path))
                    count += 1
                except Exception as e:
                    print(f"  Warning: {blob_path}: {e}")
        print(f"  {split}: {count} images from {len(session_ids)} sessions")
        if logger:
            logger.report_single_value(f"user_data/{split}", count)
            logger.report_single_value(f"total/{split}", orig_counts[split] + count)

    return user_data_dir


# ── Steps 3-5: per-model training (run in parallel on ClearML) ─────────────────

@PipelineDecorator.component(
    name="train_resnet50",
    return_values=["candidate"],
    task_type=Task.TaskTypes.training,
    parents=["ingest_data"],
    execution_queue=EXECUTION_QUEUE,
)
def component_train_resnet50(user_data_dir: str) -> dict:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    from clearml import Task as ClearMLTask
    from pipeline_helpers import train_single_model

    task = ClearMLTask.current_task()
    logger = task.get_logger() if task else None
    return train_single_model("resnet50_layer3", user_data_dir, logger)


@PipelineDecorator.component(
    name="train_mobilenet",
    return_values=["candidate"],
    task_type=Task.TaskTypes.training,
    parents=["ingest_data"],
    execution_queue=EXECUTION_QUEUE,
)
def component_train_mobilenet(user_data_dir: str) -> dict:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    from clearml import Task as ClearMLTask
    from pipeline_helpers import train_single_model

    task = ClearMLTask.current_task()
    logger = task.get_logger() if task else None
    return train_single_model("mobilenet_v3_large", user_data_dir, logger)


@PipelineDecorator.component(
    name="train_efficientnet",
    return_values=["candidate"],
    task_type=Task.TaskTypes.training,
    parents=["ingest_data"],
    execution_queue=EXECUTION_QUEUE,
)
def component_train_efficientnet(user_data_dir: str) -> dict:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    from clearml import Task as ClearMLTask
    from pipeline_helpers import train_single_model

    task = ClearMLTask.current_task()
    logger = task.get_logger() if task else None
    return train_single_model("efficientnet_b0", user_data_dir, logger)


# ── Step 6: eval_model ─────────────────────────────────────────────────────────

@PipelineDecorator.component(
    name="eval_model",
    return_values=["winner"],
    task_type=Task.TaskTypes.qc,
    parents=["train_resnet50", "train_mobilenet", "train_efficientnet"],
    execution_queue=EXECUTION_QUEUE,
)
def component_eval_model(
    candidate_resnet: dict,
    candidate_mobilenet: dict,
    candidate_efficientnet: dict,
    user_data_dir: str,
) -> dict:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    import torch
    import torch.nn as nn
    from clearml import Task as ClearMLTask
    from pipeline_helpers import build_loaders, build_model_for_eval, eval_with_metrics
    from runtime_config import get_runtime_device

    candidates = [candidate_resnet, candidate_mobilenet, candidate_efficientnet]

    device = get_runtime_device()
    criterion = nn.CrossEntropyLoss()
    _, val_loader, _ = build_loaders(user_data_dir)

    task = ClearMLTask.current_task()
    logger = task.get_logger() if task else None

    best = None
    print("Evaluating candidates on validation set:")
    for candidate in candidates:
        ckpt = torch.load(candidate["checkpoint_path"], map_location=device)
        model = build_model_for_eval(ckpt["config"], device)
        model.load_state_dict(ckpt["model_state_dict"])
        _, val_acc, fps, _ = eval_with_metrics(
            model, val_loader, criterion, device,
            name=candidate["config_name"], logger=logger,
        )
        if logger:
            logger.report_single_value(f"eval_val_acc/{candidate['config_name']}", val_acc)
            logger.report_single_value(f"eval_fps/{candidate['config_name']}", fps)
        if best is None or val_acc > best["val_acc"]:
            best = {**candidate, "val_acc": val_acc}

    print(f"Winner: {best['config_name']} (val_acc={best['val_acc']:.4f})")
    return best


# ── Step 7: get_test_result ────────────────────────────────────────────────────

@PipelineDecorator.component(
    name="get_test_result",
    return_values=["winner_test_acc"],
    task_type=Task.TaskTypes.qc,
    parents=["eval_model"],
    execution_queue=EXECUTION_QUEUE,
)
def component_get_test_result(winner: dict, user_data_dir: str) -> float:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    import torch
    import torch.nn as nn
    from clearml import Task as ClearMLTask
    from pipeline_helpers import build_loaders, build_model_for_eval, eval_with_metrics
    from runtime_config import get_runtime_device

    device = get_runtime_device()
    criterion = nn.CrossEntropyLoss()
    _, _, test_loader = build_loaders(user_data_dir)

    task = ClearMLTask.current_task()
    logger = task.get_logger() if task else None

    ckpt = torch.load(winner["checkpoint_path"], map_location=device)
    model = build_model_for_eval(ckpt["config"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc, fps, _ = eval_with_metrics(
        model, test_loader, criterion, device,
        name=f"test/{winner['config_name']}", logger=logger,
    )

    print(f"Winner ({winner['config_name']}) test acc: {test_acc:.4f} | fps: {fps:.1f}")
    if logger:
        logger.report_single_value("winner_test_acc", test_acc)
        logger.report_single_value("winner_fps", fps)

    return test_acc


# ── Step 8: evaluate_promotion ─────────────────────────────────────────────────

@PipelineDecorator.component(
    name="evaluate_promotion",
    return_values=["promoted", "prod_test_acc"],
    task_type=Task.TaskTypes.qc,
    parents=["get_test_result"],
    execution_queue=EXECUTION_QUEUE,
)
def component_evaluate_promotion(winner: dict, winner_test_acc: float, user_data_dir: str) -> tuple:
    import io, os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    import torch
    import torch.nn as nn
    from clearml import Task as ClearMLTask
    from pipeline_helpers import build_loaders, eval_loader

    from azure_storage import download_backbone
    from model import GazeClassifier, resnet50
    from runtime_config import get_runtime_device
    from training_config import NUM_CLASSES

    PROMOTION_MIN_DELTA = 0.02

    device = get_runtime_device()
    criterion = nn.CrossEntropyLoss()
    _, _, test_loader = build_loaders(user_data_dir)

    production_bytes = download_backbone()
    ckpt = torch.load(io.BytesIO(production_bytes), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    saved_cfg = ckpt.get("exp_cfg", {})
    prod_model = GazeClassifier(
        backbone=resnet50(),
        num_classes=NUM_CLASSES,
        head_dense_units=saved_cfg.get("head_dense_units", []),
        dropout=saved_cfg.get("dropout", 0.1),
        batch_norm_in_head=saved_cfg.get("batch_norm_in_head", True),
    ).to(device)
    prod_model.load_state_dict(state_dict)
    _, prod_test_acc = eval_loader(prod_model, test_loader, criterion, device)

    promoted = winner_test_acc >= prod_test_acc + PROMOTION_MIN_DELTA
    print(f"Candidate: {winner_test_acc:.4f} | Production: {prod_test_acc:.4f} | Need +{PROMOTION_MIN_DELTA} → promoted={promoted}")

    task = ClearMLTask.current_task()
    if task:
        logger = task.get_logger()
        logger.report_single_value("production_test_acc", prod_test_acc)
        logger.report_single_value("promoted", int(promoted))

    return promoted, prod_test_acc


# ── Step 9: upload_promoted ────────────────────────────────────────────────────

@PipelineDecorator.component(
    name="upload_promoted",
    return_values=[],
    task_type=Task.TaskTypes.custom,
    parents=["evaluate_promotion"],
    execution_queue=EXECUTION_QUEUE,
)
def component_upload_promoted(winner: dict, winner_test_acc: float, prod_test_acc: float) -> None:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    from datetime import datetime, timezone
    from azure_storage import (
        load_manifest, load_model_version, save_last_retrain_info,
        save_model_version, upload_bytes,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    version_info = load_model_version()
    next_version = version_info["latest_version"] + 1
    promoted_blob = f"models/backbone/promoted/ethxgaze_v{next_version}_{timestamp}.pth"

    with open(winner["checkpoint_path"], "rb") as f:
        model_bytes = f.read()

    upload_bytes(promoted_blob, model_bytes)
    upload_bytes("models/backbone/ethxgaze_backbone.pth", model_bytes)
    save_model_version({
        "latest_version": next_version,
        "latest_promoted_blob": promoted_blob,
        "latest_promoted_timestamp": timestamp,
        "latest_candidate_test_acc": winner_test_acc,
        "latest_production_test_acc": prod_test_acc,
    })
    manifest = load_manifest()
    train_entries = [e for e in manifest if e.get("split") == "train"]
    save_last_retrain_info({
        "timestamp": timestamp,
        "num_train_sessions": len(train_entries),
        "num_train_samples": sum(e.get("num_samples", 0) for e in train_entries),
    })
    print(f"Promoted to v{next_version}: {promoted_blob}")


# ── Pipeline definition ────────────────────────────────────────────────────────

@PipelineDecorator.pipeline(
    name=PIPELINE_NAME,
    project=CLEARML_PROJECT,
    version="3.0",
    pipeline_execution_queue=CONTROLLER_QUEUE,
    add_pipeline_tags=True,
)
def retrain_pipeline(force: bool = False) -> dict:
    try:
        should_retrain, train_ids, val_ids, test_ids = component_check_retrain(force=force)

        if not should_retrain:
            print("Volume threshold not met — skipping retrain.")
            return {"status": "skipped"}

        user_data_dir = component_ingest_data(train_ids, val_ids, test_ids)

        # These three steps are independent — ClearML schedules them in parallel
        candidate_resnet = component_train_resnet50(user_data_dir)
        candidate_mobilenet = component_train_mobilenet(user_data_dir)
        candidate_efficientnet = component_train_efficientnet(user_data_dir)

        winner = component_eval_model(
            candidate_resnet, candidate_mobilenet, candidate_efficientnet,
            user_data_dir,
        )
        winner_test_acc = component_get_test_result(winner, user_data_dir)
        promoted, prod_test_acc = component_evaluate_promotion(winner, winner_test_acc, user_data_dir)

        if promoted:
            component_upload_promoted(winner, winner_test_acc, prod_test_acc)

        print(f"\nPipeline complete | winner={winner['config_name']} | promoted={promoted} | test_acc={float(winner_test_acc):.4f}")
        return {"status": "done", "promoted": promoted, "winner": winner["config_name"], "test_acc": winner_test_acc}

    finally:
        shutil.rmtree(PIPELINE_TMP_DIR, ignore_errors=True)
        print(f"Cleaned up {PIPELINE_TMP_DIR}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EyeSist retrain pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-local", action="store_true",
                      help="Run all steps locally in this process (used by cron job)")
    mode.add_argument("--run-now", action="store_true",
                      help="Trigger on ClearML agents (NOTE: local /tmp/ paths not shared across agents)")
    parser.add_argument("--force", action="store_true",
                        help="Skip volume threshold check and run regardless")
    args = parser.parse_args()

    if args.run_local:
        PipelineDecorator.run_locally()
        retrain_pipeline(force=args.force)
    elif args.run_now:
        retrain_pipeline(force=args.force)
