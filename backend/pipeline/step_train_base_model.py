"""
Retrain the EyeSist GazeClassifier starting from the current production model.

Starting weights : production model (ethxgaze_backbone.pth on Azure) — NOT epoch_24_ckpt.pth.tar.
Train set        : local TRAIN_DIR      +  Azure user sessions in train_session_ids.
Val set          : local VAL_DIR        +  Azure user sessions in val_session_ids.
Output           : candidate .pth uploaded to Azure; returns {candidate_blob, val_accuracy, ...}.

Designed to run as a ClearML pipeline component (Task already initialised by the pipeline).
Can also be run standalone: python step_train_base_model.py --session-ids sid1 sid2 ...
"""

import copy
import io
import os
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms

from azure_storage import (
    download_backbone,
    download_bytes,
    list_session_images,
    upload_bytes,
)
from model import GazeClassifier, _ResizeWithPad
from runtime_config import get_runtime_device, get_runtime_device_str
from training_config import (
    BATCH_SIZE,
    DATA_DIR,
    IMG_SIZE,
    LABELS,
    NUM_CLASSES,
    NUM_WORKERS,
    RETRAIN_EXPERIMENT_CONFIG,
    TRAIN_DIR,
    VAL_DIR,
    seed_everything,
)

CANDIDATE_BLOB_PREFIX = "models/backbone/candidate"


# ── Dataset helpers ────────────────────────────────────────────────────────────

class _UserSessionDataset(Dataset):
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


def _download_sessions(session_ids: list[str]) -> list[tuple[bytes, int]]:
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    samples: list[tuple[bytes, int]] = []
    for sid in session_ids:
        for blob_path in list_session_images(sid):
            label = _label_from_blob_name(blob_path)
            if label is None:
                continue
            try:
                samples.append((download_bytes(blob_path), label_to_idx[label]))
            except Exception as e:
                print(f"  Warning: skipping {blob_path}: {e}")
    print(f"Downloaded {len(samples)} user images from {len(session_ids)} sessions")
    return samples


def _build_transforms(img_size: int = IMG_SIZE):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose([
        _ResizeWithPad(img_size, fill=0),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        _ResizeWithPad(img_size, fill=0),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def _ensure_local_dataset_dirs(*paths: str) -> None:
    missing_paths = [path for path in paths if not os.path.isdir(path)]
    if not missing_paths:
        return

    dataset_path = os.path.abspath(DATA_DIR)
    print(f"Local dataset not found. Please download dataset in {dataset_path} locally")
    raise FileNotFoundError(f"Local dataset not found. Expected directories: {missing_paths}")


def _build_dataloaders(
    user_train_samples: list[tuple[bytes, int]],
    user_val_samples: list[tuple[bytes, int]],
):
    train_tf, eval_tf = _build_transforms()
    _ensure_local_dataset_dirs(TRAIN_DIR, VAL_DIR)

    original_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    original_val = datasets.ImageFolder(VAL_DIR, transform=eval_tf)

    # Train: original + user train sessions
    if user_train_samples:
        user_train_ds = _UserSessionDataset(user_train_samples, train_tf)
        train_dataset = ConcatDataset([original_train, user_train_ds])
        print(f"Train: {len(original_train)} original + {len(user_train_ds)} user = {len(train_dataset)} total")
    else:
        train_dataset = original_train
        print(f"Train: {len(original_train)} original (no user train data)")

    # Val: original + user val sessions
    if user_val_samples:
        user_val_ds = _UserSessionDataset(user_val_samples, eval_tf)
        val_dataset = ConcatDataset([original_val, user_val_ds])
        print(f"Val:   {len(original_val)} original + {len(user_val_ds)} user = {len(val_dataset)} total")
    else:
        val_dataset = original_val
        print(f"Val:   {len(original_val)} original (no user val data)")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader


# ── Model freeze / unfreeze ────────────────────────────────────────────────────

def _unfreeze_backbone_from(
    model: GazeClassifier, layer_name: str, unfreeze_batchnorm: bool = False
) -> None:
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
        f"{trainable:,} / {total:,} trainable ({100 * trainable / total:.1f}%)"
    )


# ── Training loop ──────────────────────────────────────────────────────────────

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


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        out = model(images)
        loss_sum += criterion(out, targets).item() * images.size(0)
        correct += out.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return loss_sum / total, correct / total


class _EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_score: float | None = None
        self.best_state = None
        self.should_stop = False

    def __call__(self, val_acc: float, model):
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


def _train_phase(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    num_epochs: int, device, phase_name: str, patience: int,
    logger=None, epoch_offset: int = 0,
) -> list[dict]:
    stopper = _EarlyStopping(patience)
    history = []
    for epoch in range(num_epochs):
        g = epoch_offset + epoch + 1
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _evaluate(model, val_loader, criterion, device)

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(va_loss)
        elif scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{phase_name}] {epoch+1}/{num_epochs} (g{g}) | "
            f"train {tr_acc:.4f} | val {va_acc:.4f} | lr {lr:.2e}"
        )

        if logger is not None:
            logger.report_scalar("loss", "train", tr_loss, iteration=g)
            logger.report_scalar("loss", "val", va_loss, iteration=g)
            logger.report_scalar("accuracy", "train", tr_acc, iteration=g)
            logger.report_scalar("accuracy", "val", va_acc, iteration=g)
            logger.report_scalar("lr", phase_name, lr, iteration=g)

        history.append({
            "phase": phase_name, "epoch_global": g,
            "loss": tr_loss, "accuracy": tr_acc,
            "val_loss": va_loss, "val_accuracy": va_acc, "lr": lr,
        })
        stopper(va_acc, model)
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    stopper.restore_best(model)
    return history


# ── Entry point ────────────────────────────────────────────────────────────────

def run_training(train_session_ids: list[str], val_session_ids: list[str]) -> dict:
    """
    Retrain step — designed to run as a ClearML pipeline component.

    Args:
        train_session_ids: session IDs with split='train' from the manifest.
        val_session_ids:   session IDs with split='val'   from the manifest.

    Returns:
        {candidate_blob, val_accuracy, timestamp, num_train_samples, num_val_samples}
    """
    seed_everything()
    cfg = RETRAIN_EXPERIMENT_CONFIG
    device = get_runtime_device()
    print(f"Training step running on device: {get_runtime_device_str()}")
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

    # Download user session images for both splits
    user_train_samples = _download_sessions(train_session_ids)
    user_val_samples   = _download_sessions(val_session_ids)
    train_loader, val_loader = _build_dataloaders(user_train_samples, user_val_samples)

    # Load production model as starting point
    print("Loading production model from Azure")
    model_bytes = download_backbone()
    checkpoint = torch.load(io.BytesIO(model_bytes), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cfg_saved = checkpoint.get("exp_cfg", cfg)
    from model import resnet50 as _resnet50
    _backbone = _resnet50()
    model = GazeClassifier(
        backbone=_backbone,
        num_classes=NUM_CLASSES,
        head_dense_units=cfg_saved.get("head_dense_units", cfg["head_dense_units"]),
        dropout=cfg_saved.get("dropout", cfg["dropout"]),
        batch_norm_in_head=cfg_saved.get("batch_norm_in_head", cfg["batch_norm_in_head"]),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    patience = cfg.get("early_stopping_patience", 10)
    all_history: list[dict] = []

    # Phase 1 — retrain head with backbone frozen
    print("\n--- Phase 1: head only ---")
    model.freeze_backbone()
    opt1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["phase1_lr"],
    )
    sched1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, mode="min", factor=0.5, patience=2)
    all_history.extend(
        _train_phase(
            model, train_loader, val_loader, criterion, opt1, sched1,
            num_epochs=cfg["phase1_epochs"], device=device, phase_name="phase1",
            patience=patience, logger=logger, epoch_offset=0,
        )
    )

    # Phase 2 — unfreeze upper backbone layers and fine-tune
    if cfg.get("phase2_epochs", 0) > 0:
        print(f"\n--- Phase 2: fine-tune from '{cfg['unfreeze_from_layer']}' ---")
        _unfreeze_backbone_from(
            model,
            layer_name=cfg["unfreeze_from_layer"],
            unfreeze_batchnorm=cfg.get("unfreeze_batchnorm", False),
        )
        backbone_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("backbone")
        ]
        head_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("backbone")
        ]
        opt2 = optim.Adam([
            {"params": backbone_params, "lr": cfg["phase2_lr"] * 0.1},
            {"params": head_params, "lr": cfg["phase2_lr"]},
        ])
        sched2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=3)
        all_history.extend(
            _train_phase(
                model, train_loader, val_loader, criterion, opt2, sched2,
                num_epochs=cfg["phase2_epochs"], device=device, phase_name="phase2",
                patience=patience, logger=logger, epoch_offset=len(all_history),
            )
        )

    best_val_acc = max(h["val_accuracy"] for h in all_history) if all_history else 0.0
    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # Upload candidate model to Azure
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

    return {
        "candidate_blob": candidate_blob,
        "val_accuracy": best_val_acc,
        "timestamp": timestamp,
        "num_train_samples": len(user_train_samples),
        "num_val_samples": len(user_val_samples),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-ids", nargs="*", default=[], help="Train session IDs")
    args = parser.parse_args()
    print(run_training(train_session_ids=args.session_ids))
