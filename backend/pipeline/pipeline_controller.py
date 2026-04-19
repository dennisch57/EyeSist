"""
EyeSist base model retrain pipeline.

All step logic lives in this single file. Each step function is self-contained:
local imports happen inside the function body so ClearML remote agents don't
need the pipeline package on their path. Steps are wired together with
PipelineController.add_function_step().

Pipeline steps:
  1. check_retrain    — volume-based gate; assigns 70/15/15 splits via greedy balancer
  2. train_model      — retrain from production weights on full train split
  3. evaluate_promote — compare candidate vs production on frozen cumulative test set

Usage:
  Register weekly schedule (run once to activate):
    python pipeline_controller.py --schedule

  Trigger immediately (manual override, bypasses schedule):
    python pipeline_controller.py --run-now

  Local debug (all steps run in this process, no ClearML agents needed):
    python pipeline_controller.py --run-local
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from clearml.automation import PipelineController

CLEARML_PROJECT = "EyeSist"
PIPELINE_NAME = "Base Model Retrain"
CONTROLLER_QUEUE = "services"
EXECUTION_QUEUE = "default"
WEEKLY_CRON = "0 2 * * 1"  # 2am UTC every Monday


# ── Module-level helpers (no local imports — safe to pass via helper_functions) ─

def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        correct += out.detach().argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return loss_sum / total, correct / total


def _evaluate_loader(model, loader, criterion, device):
    import torch
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            out = model(images)
            loss_sum += criterion(out, targets).item() * images.size(0)
            correct += out.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return loss_sum / total, correct / total


def _evaluate_acc(model, loader, device):
    import torch
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            correct += model(images).argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


def _unfreeze_backbone_from(model, layer_name: str, unfreeze_batchnorm: bool = False) -> None:
    import torch.nn as nn
    layer_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "avgpool"]
    if layer_name not in layer_order:
        raise ValueError(f"layer_name must be one of {layer_order}, got '{layer_name}'")
    unfreeze_idx = layer_order.index(layer_name)
    for param in model.backbone.parameters():
        param.requires_grad = False
    for name, module in model.backbone.named_children():
        if name in layer_order and layer_order.index(name) >= unfreeze_idx:
            for param in module.parameters():
                param.requires_grad = True
            if not unfreeze_batchnorm:
                for m in module.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Unfrozen from '{layer_name}' (BN frozen: {not unfreeze_batchnorm}): "
        f"{trainable:,}/{total:,} trainable ({100 * trainable / total:.1f}%)"
    )


# ── Step 1: check_retrain ──────────────────────────────────────────────────────

def step_check_retrain() -> tuple[bool, list, list]:
    import subprocess, sys, os
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "azure-identity", "azure-storage-blob", "python-dotenv"])
    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))

    from azure_storage import load_manifest, load_last_retrain_info
    from training_config import RETRAIN_NEW_SAMPLES_THRESHOLD

    manifest = load_manifest()
    last = load_last_retrain_info()

    train_entries = [e for e in manifest if e.get("split") == "train"]
    val_entries   = [e for e in manifest if e.get("split") == "val"]

    current_train_sessions = len(train_entries)
    current_train_samples  = sum(e.get("num_samples", 0) for e in train_entries)
    prev_train_sessions    = last.get("num_train_sessions", 0)
    prev_train_samples     = last.get("num_train_samples", 0)

    new_sessions = current_train_sessions - prev_train_sessions
    new_samples  = current_train_samples  - prev_train_samples

    print(f"Train pool : {current_train_sessions} sessions / {current_train_samples} samples")
    print(f"Since last retrain: +{new_sessions} sessions / +{new_samples} samples")
    print(f"Threshold  : {RETRAIN_NEW_SAMPLES_THRESHOLD} samples")

    should_retrain = True
    # TODO: Uncomment after testing

    #should_retrain = new_samples >= RETRAIN_NEW_SAMPLES_THRESHOLD

    # if not should_retrain:
    #     print("Volume threshold not met — skipping retrain.")
    #     return False, [], []

    train_ids = [e["session_id"] for e in train_entries]
    val_ids   = [e["session_id"] for e in val_entries]

    print(f"Retrain triggered: {len(train_ids)} train, {len(val_ids)} val sessions")
    return should_retrain, train_ids, val_ids


# ── Step 2: train_model ────────────────────────────────────────────────────────

def step_train_model(
    should_retrain: bool, train_session_ids: list, val_session_ids: list
) -> tuple[str, float]:
    if not should_retrain:
        print("Volume threshold not met — skipping training.")
        return "", 0.0

    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "azure-identity", "azure-storage-blob",
                           "torch", "torchvision", "pillow", "numpy", "python-dotenv"])
    import copy, io, os
    sys.path.insert(0, os.getcwd())
    from datetime import datetime, timezone
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import ConcatDataset, DataLoader, Dataset
    from torchvision import datasets, transforms

    from azure_storage import download_backbone, download_bytes, list_session_images, upload_bytes
    from model import GazeClassifier, _ResizeWithPad, resnet50
    from runtime_config import get_runtime_device, get_runtime_device_str
    from training_config import (
        BATCH_SIZE, DATA_DIR, IMG_SIZE, LABELS, NUM_CLASSES, NUM_WORKERS,
        RETRAIN_EXPERIMENT_CONFIG, TRAIN_DIR, VAL_DIR, seed_everything,
    )

    CANDIDATE_BLOB_PREFIX = "models/backbone/candidate"
    label_to_idx = {label: i for i, label in enumerate(LABELS)}

    class _UserSessionDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_bytes, label_idx = self.samples[idx]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return self.transform(img), label_idx

    class _EarlyStopping:
        def __init__(self, patience):
            self.patience = patience
            self.counter = 0
            self.best_score = None
            self.best_state = None
            self.should_stop = False

        def __call__(self, val_acc, model):
            if self.best_score is None or val_acc > self.best_score:
                self.best_score = val_acc
                self.best_state = copy.deepcopy(model.state_dict())
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True

        def restore_best(self, model):
            if self.best_state is not None:
                model.load_state_dict(self.best_state)
                print(f"Restored best checkpoint (val_acc: {self.best_score:.4f})")

    def _download_sessions(session_ids):
        samples = []
        for sid in session_ids:
            for blob_path in list_session_images(sid):
                parts = blob_path.split("/")
                label = parts[-2] if len(parts) >= 2 else None
                if label not in LABELS:
                    continue
                try:
                    samples.append((download_bytes(blob_path), label_to_idx[label]))
                except Exception as e:
                    print(f"  Warning: skipping {blob_path}: {e}")
        print(f"Downloaded {len(samples)} user images from {len(session_ids)} sessions")
        return samples

    seed_everything()
    cfg = RETRAIN_EXPERIMENT_CONFIG
    device = get_runtime_device()
    print(f"Training on: {get_runtime_device_str()}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    from clearml import Task
    task = Task.current_task()
    if task is None:
        task = Task.init(
            project_name="EyeSist",
            task_name=f"train_base_model_{timestamp}",
            task_type=Task.TaskTypes.training,
            reuse_last_task_id=False,
        )
    logger = task.get_logger()
    task.connect(cfg, name="retrain_config")
    task.connect(
        {"train_session_ids": train_session_ids, "val_session_ids": val_session_ids},
        name="input_data",
    )

    user_train_samples = _download_sessions(train_session_ids)
    user_val_samples   = _download_sessions(val_session_ids)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        _ResizeWithPad(IMG_SIZE, fill=0),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), normalize,
    ])
    eval_tf = transforms.Compose([
        _ResizeWithPad(IMG_SIZE, fill=0), transforms.ToTensor(), normalize,
    ])

    if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(VAL_DIR)):
        raise FileNotFoundError(f"Local dataset not found at {os.path.abspath(DATA_DIR)}")

    original_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    original_val   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)

    train_ds = (
        ConcatDataset([original_train, _UserSessionDataset(user_train_samples, train_tf)])
        if user_train_samples else original_train
    )
    val_ds = (
        ConcatDataset([original_val, _UserSessionDataset(user_val_samples, eval_tf)])
        if user_val_samples else original_val
    )
    print(f"Train: {len(train_ds)} total | Val: {len(val_ds)} total")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print("Loading production model from Azure")
    model_bytes = download_backbone()
    checkpoint  = torch.load(io.BytesIO(model_bytes), map_location=device)
    state_dict  = checkpoint.get("model_state_dict", checkpoint)
    cfg_saved   = checkpoint.get("exp_cfg", cfg)
    model = GazeClassifier(
        backbone=resnet50(),
        num_classes=NUM_CLASSES,
        head_dense_units=cfg_saved.get("head_dense_units", cfg["head_dense_units"]),
        dropout=cfg_saved.get("dropout", cfg["dropout"]),
        batch_norm_in_head=cfg_saved.get("batch_norm_in_head", cfg["batch_norm_in_head"]),
    )
    model.load_state_dict(state_dict)
    model.to(device)

    criterion  = nn.CrossEntropyLoss()
    patience   = cfg.get("early_stopping_patience", 10)
    all_history: list[dict] = []

    def _run_phase(phase_name, num_epochs, optimizer, scheduler, epoch_offset=0):
        stopper = _EarlyStopping(patience)
        for epoch in range(num_epochs):
            g = epoch_offset + epoch + 1
            tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_acc = _evaluate_loader(model, val_loader, criterion, device)
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss)
            elif scheduler is not None:
                scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[{phase_name}] {epoch+1}/{num_epochs} (g{g}) | "
                f"train {tr_acc:.4f} | val {va_acc:.4f} | lr {lr:.2e}"
            )
            logger.report_scalar("loss",     "train", tr_loss, iteration=g)
            logger.report_scalar("loss",     "val",   va_loss, iteration=g)
            logger.report_scalar("accuracy", "train", tr_acc,  iteration=g)
            logger.report_scalar("accuracy", "val",   va_acc,  iteration=g)
            logger.report_scalar("lr", phase_name, lr, iteration=g)
            all_history.append({
                "phase": phase_name, "epoch_global": g,
                "loss": tr_loss, "accuracy": tr_acc,
                "val_loss": va_loss, "val_accuracy": va_acc, "lr": lr,
            })
            stopper(va_acc, model)
            if stopper.should_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        stopper.restore_best(model)

    print("\n--- Phase 1: head only ---")
    model.freeze_backbone()
    opt1   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["phase1_lr"])
    sched1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, mode="min", factor=0.5, patience=2)
    _run_phase("phase1", cfg["phase1_epochs"], opt1, sched1, epoch_offset=0)

    if cfg.get("phase2_epochs", 0) > 0:
        print(f"\n--- Phase 2: fine-tune from '{cfg['unfreeze_from_layer']}' ---")
        _unfreeze_backbone_from(model, cfg["unfreeze_from_layer"], cfg.get("unfreeze_batchnorm", False))
        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone")]
        head_params     = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")]
        opt2 = optim.Adam([
            {"params": backbone_params, "lr": cfg["phase2_lr"] * 0.1},
            {"params": head_params,     "lr": cfg["phase2_lr"]},
        ])
        sched2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=3)
        _run_phase("phase2", cfg["phase2_epochs"], opt2, sched2, epoch_offset=len(all_history))

    best_val_acc = max(h["val_accuracy"] for h in all_history) if all_history else 0.0
    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    candidate_blob = f"{CANDIDATE_BLOB_PREFIX}/ethxgaze_candidate_{timestamp}.pth"
    buf = io.BytesIO()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "exp_cfg": cfg,
            "labels": LABELS,
            "val_accuracy": best_val_acc,
            "timestamp": timestamp,
        },
        buf,
    )
    upload_bytes(candidate_blob, buf.getvalue())
    print(f"Candidate uploaded: {candidate_blob}")

    logger.report_single_value("best_val_accuracy", best_val_acc)
    task.upload_artifact("candidate_blob_path", candidate_blob)

    return candidate_blob, best_val_acc


# ── Step 3: evaluate_promote ───────────────────────────────────────────────────

def step_evaluate_promote(
    should_retrain: bool, candidate_blob: str, candidate_val_accuracy: float
) -> tuple[bool, float]:
    if not should_retrain or not candidate_blob:
        print("Skipping evaluation — no candidate to promote.")
        return False, 0.0

    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "azure-identity", "azure-storage-blob",
                           "torch", "torchvision", "pillow", "numpy", "python-dotenv"])
    import io, os
    sys.path.insert(0, os.getcwd())
    from datetime import datetime, timezone
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))

    import torch
    from PIL import Image
    from torch.utils.data import ConcatDataset, DataLoader, Dataset
    from torchvision import datasets, transforms

    from azure_storage import (
        delete_blob, download_backbone, download_bytes, download_json,
        list_session_images, load_manifest, load_model_version,
        save_last_retrain_info, save_model_version, upload_bytes, upload_json,
    )
    from model import GazeClassifier, _ResizeWithPad, resnet50
    from runtime_config import get_runtime_device, get_runtime_device_str
    from training_config import (
        BATCH_SIZE, DATA_DIR, IMG_SIZE, LABELS, NUM_CLASSES, NUM_WORKERS,
        RETRAIN_EXPERIMENT_CONFIG, TEST_DIR,
    )

    PROMOTION_MIN_DELTA = 0.05
    PROMOTION_LOG_BLOB  = "models/backbone/promotion_log.json"
    label_to_idx = {label: i for i, label in enumerate(LABELS)}

    class _UserTestDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_bytes, label_idx = self.samples[idx]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return self.transform(img), label_idx

    device = get_runtime_device()
    print(f"Evaluation on: {get_runtime_device_str()}")
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

    manifest  = load_manifest()
    test_ids  = [e["session_id"] for e in manifest if e.get("split") == "test"]
    print(f"Found {len(test_ids)} test-split sessions in manifest")

    user_samples: list[tuple[bytes, int]] = []
    for sid in test_ids:
        for blob_path in list_session_images(sid):
            parts = blob_path.split("/")
            label = parts[-2] if len(parts) >= 2 else None
            if label not in LABELS:
                continue
            try:
                user_samples.append((download_bytes(blob_path), label_to_idx[label]))
            except Exception as e:
                print(f"  Warning: skipping {blob_path}: {e}")
    print(f"Downloaded {len(user_samples)} user test images")

    eval_tf = transforms.Compose([
        _ResizeWithPad(IMG_SIZE, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Local dataset not found at {os.path.abspath(DATA_DIR)}")
    original_test = datasets.ImageFolder(TEST_DIR, transform=eval_tf)
    test_ds = (
        ConcatDataset([original_test, _UserTestDataset(user_samples, eval_tf)])
        if user_samples else original_test
    )
    print(f"Test: {len(original_test)} original + {len(user_samples)} user = {len(test_ds)} total")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    def _load_model(model_bytes):
        checkpoint = torch.load(io.BytesIO(model_bytes), map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cfg = checkpoint.get("exp_cfg", RETRAIN_EXPERIMENT_CONFIG)
        m = GazeClassifier(
            backbone=resnet50(),
            num_classes=NUM_CLASSES,
            head_dense_units=cfg.get("head_dense_units", []),
            dropout=cfg.get("dropout", 0.1),
            batch_norm_in_head=cfg.get("batch_norm_in_head", True),
        )
        m.load_state_dict(state_dict)
        m.to(device)
        m.eval()
        return m

    print("Loading candidate model from Azure")
    candidate_bytes = download_bytes(candidate_blob)
    candidate_model = _load_model(candidate_bytes)

    print("Loading production model from Azure")
    production_model = _load_model(download_backbone())

    candidate_test_acc  = _evaluate_acc(candidate_model, test_loader, device)
    production_test_acc = _evaluate_acc(production_model, test_loader, device)

    print(f"Candidate  test acc: {candidate_test_acc:.4f}  (val: {candidate_val_accuracy:.4f})")
    print(f"Production test acc: {production_test_acc:.4f}")

    logger.report_single_value("candidate_val_accuracy",  candidate_val_accuracy)
    logger.report_single_value("candidate_test_accuracy", candidate_test_acc)
    logger.report_single_value("production_test_accuracy", production_test_acc)

    promoted      = candidate_test_acc >= production_test_acc + PROMOTION_MIN_DELTA
    promoted_blob = None

    if promoted:
        version_info  = load_model_version()
        next_version  = version_info["latest_version"] + 1
        promoted_blob = f"models/backbone/promoted/ethxgaze_v{next_version}_{timestamp}.pth"
        print(
            f"Promoting candidate → v{next_version} "
            f"({candidate_test_acc:.4f} > {production_test_acc:.4f} + {PROMOTION_MIN_DELTA})"
        )
        upload_bytes(promoted_blob, candidate_bytes)
        save_model_version({
            "latest_version": next_version,
            "latest_promoted_blob": promoted_blob,
            "latest_promoted_timestamp": timestamp,
            "latest_candidate_test_acc": candidate_test_acc,
            "latest_production_test_acc": production_test_acc,
            "production_blob": version_info.get("production_blob", "models/backbone/ethxgaze_backbone.pth"),
            "production_version": version_info.get("production_version"),
        })
        train_entries = [e for e in manifest if e.get("split") == "train"]
        save_last_retrain_info({
            "timestamp": timestamp,
            "num_train_sessions": len(train_entries),
            "num_train_samples": sum(e.get("num_samples", 0) for e in train_entries),
        })
        print(f"Candidate approved for manual rollout → {promoted_blob}")
    else:
        print(
            f"NOT promoted — need +{PROMOTION_MIN_DELTA} improvement "
            f"({candidate_test_acc:.4f} vs {production_test_acc:.4f})."
        )

    try:
        delete_blob(candidate_blob)
        print(f"Deleted candidate blob: {candidate_blob}")
    except Exception as e:
        print(f"Warning: could not delete candidate blob: {e}")

    promotion_record = {
        "timestamp": timestamp,
        "promoted": promoted,
        "candidate_val_accuracy": candidate_val_accuracy,
        "candidate_test_accuracy": candidate_test_acc,
        "production_test_accuracy": production_test_acc,
        "promoted_blob": promoted_blob,
    }
    try:
        existing = []
        try:
            existing = download_json(PROMOTION_LOG_BLOB)
        except Exception:
            pass
        existing.append(promotion_record)
        upload_json(PROMOTION_LOG_BLOB, existing)
    except Exception as e:
        print(f"Warning: could not update promotion log: {e}")

    logger.report_single_value("promoted", int(promoted))
    task.upload_artifact("promotion_record", promotion_record)

    return promoted, candidate_test_acc


# ── Pipeline definition ────────────────────────────────────────────────────────

def build_and_run_pipeline(run_locally: bool = False) -> None:
    pipe = PipelineController(
        name=PIPELINE_NAME,
        project=CLEARML_PROJECT,
        version="1.0",
        add_pipeline_tags=True,
        abort_on_failure=False,
    )
    pipe.set_default_execution_queue(EXECUTION_QUEUE)

    pipe.add_function_step(
        name="check_retrain",
        function=step_check_retrain,
        function_return=["should_retrain", "train_session_ids", "val_session_ids"],
        packages=["azure-identity", "azure-storage-blob", "python-dotenv"],
        task_type="data_processing",
        working_dir="backend",
    )

    pipe.add_function_step(
        name="train_model",
        parents=["check_retrain"],
        function=step_train_model,
        function_kwargs=dict(
            should_retrain="${check_retrain.should_retrain}",
            train_session_ids="${check_retrain.train_session_ids}",
            val_session_ids="${check_retrain.val_session_ids}",
        ),
        function_return=["candidate_blob", "val_accuracy"],
        helper_functions=[_train_one_epoch, _evaluate_loader, _unfreeze_backbone_from],
        packages=[
            "azure-identity", "azure-storage-blob",
            "torch", "torchvision", "pillow", "numpy", "scikit-learn", "python-dotenv",
        ],
        task_type="training",
        working_dir="backend",
    )

    pipe.add_function_step(
        name="evaluate_promote",
        parents=["train_model"],
        function=step_evaluate_promote,
        function_kwargs=dict(
            should_retrain="${check_retrain.should_retrain}",
            candidate_blob="${train_model.candidate_blob}",
            candidate_val_accuracy="${train_model.val_accuracy}",
        ),
        function_return=["promoted", "candidate_test_acc"],
        helper_functions=[_evaluate_acc],
        packages=[
            "azure-identity", "azure-storage-blob",
            "torch", "torchvision", "pillow", "numpy", "python-dotenv",
        ],
        task_type="qc",
        working_dir="backend",
    )

    if run_locally:
        pipe.start_locally(run_pipeline_steps_locally=True)
    else:
        pipe.start(queue=CONTROLLER_QUEUE)


# ── Schedule registration ──────────────────────────────────────────────────────

def register_schedule(cron: str = WEEKLY_CRON) -> None:
    """
    Register this pipeline as a weekly ClearML scheduled job.
    Run this once; the ClearML scheduler service handles all future executions.
    To change the schedule, re-run with a different --cron value.
    """
    from clearml.automation.scheduler import TaskScheduler

    parts = cron.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5-field cron expression, got: '{cron}'")

    minute_s, hour_s, day_s, month_s, weekday_s = parts
    if day_s != "*" or month_s != "*":
        raise ValueError(
            "This ClearML scheduler wrapper only supports weekly cron expressions "
            "with '*' for day-of-month and month."
        )

    try:
        minute = int(minute_s)
        hour   = int(hour_s)
    except ValueError as exc:
        raise ValueError(f"Invalid cron minute/hour in '{cron}'") from exc

    weekday_map = {
        "0": "sunday", "1": "monday", "2": "tuesday", "3": "wednesday",
        "4": "thursday", "5": "friday", "6": "saturday", "7": "sunday",
    }
    weekdays = [weekday_map.get(token.strip()) for token in weekday_s.split(",")]
    if any(day is None for day in weekdays):
        raise ValueError(
            "Unsupported cron weekday field. Use comma-separated numeric weekdays "
            "like '1' for Monday."
        )

    print(f"Registering weekly pipeline schedule (cron: '{cron}')")

    #TODO: revert to 60 after testing
    scheduler = TaskScheduler(
        sync_frequency_minutes=5,  # how often the scheduler checks for pending tasks to run
    )
    scheduler.add_task(
        schedule_function=build_and_run_pipeline,
        queue=CONTROLLER_QUEUE,
        name=f"{PIPELINE_NAME} — Weekly",
        minute=minute,
        hour=hour,
        weekdays=weekdays,
        target_project=CLEARML_PROJECT,
        execute_immediately=True,
    )
    scheduler.start_remotely()
    print("Scheduler registered as a ClearML service task.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EyeSist retrain pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--schedule",
        action="store_true",
        help="Register weekly ClearML scheduler (run once to activate)",
    )
    mode.add_argument(
        "--run-now",
        action="store_true",
        help="Trigger the pipeline immediately on ClearML agents (manual override)",
    )
    mode.add_argument(
        "--run-local",
        action="store_true",
        help="Run all steps locally in this process (debug only)",
    )
    parser.add_argument(
        "--cron",
        default=WEEKLY_CRON,
        metavar="EXPR",
        help=f"Cron expression for --schedule (default: '{WEEKLY_CRON}' = 2am UTC Monday)",
    )
    args = parser.parse_args()

    if args.schedule:
        register_schedule(cron=args.cron)
    elif args.run_now:
        print("Triggering pipeline immediately on ClearML agents...")
        build_and_run_pipeline()
    elif args.run_local:
        build_and_run_pipeline(run_locally=True)
