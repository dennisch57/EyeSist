"""
Shared helpers for the EyeSist retrain pipeline steps.

Lives in backend/ so it is importable from any ClearML component step that has
backend/ on sys.path (which every component achieves via its sys.path.insert).
"""

import copy
import io
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
)

from model import GazeClassifier, _ResizeWithPad, resnet50
from runtime_config import get_runtime_device, get_runtime_device_str
from training_config import (
    BATCH_SIZE,
    IMG_SIZE,
    LABELS,
    MODEL_CONFIGS,
    NUM_CLASSES,
    NUM_WORKERS,
    PIPELINE_TMP_DIR,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
    seed_everything,
)
from azure_storage import download_backbone


# ── Dataset ────────────────────────────────────────────────────────────────────

class FolderDataset(Dataset):
    def __init__(self, root: str, transform):
        self.samples: list[tuple[str, int]] = []
        self.transform = transform
        label_to_idx = {label: i for i, label in enumerate(LABELS)}
        for label in LABELS:
            label_dir = os.path.join(root, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(label_dir, fname), label_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


# ── Transforms / loaders ───────────────────────────────────────────────────────

def build_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        _ResizeWithPad(IMG_SIZE, fill=0),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        _ResizeWithPad(IMG_SIZE, fill=0),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def build_loaders(user_data_dir: str):
    train_tf, eval_tf = build_transforms()
    loaders = {}
    for split, orig_dir, tf in [
        ("train", TRAIN_DIR, train_tf),
        ("val",   VAL_DIR,   eval_tf),
        ("test",  TEST_DIR,  eval_tf),
    ]:
        parts = []
        if os.path.isdir(orig_dir):
            parts.append(datasets.ImageFolder(orig_dir, transform=tf))
        user_split_dir = os.path.join(user_data_dir, split)
        if os.path.isdir(user_split_dir):
            user_ds = FolderDataset(user_split_dir, tf)
            if len(user_ds) > 0:
                parts.append(user_ds)
        if not parts:
            raise FileNotFoundError(
                f"No data for split '{split}' (checked {orig_dir} and {user_split_dir})"
            )
        combined = parts[0] if len(parts) == 1 else ConcatDataset(parts)
        loaders[split] = DataLoader(
            combined, batch_size=BATCH_SIZE, shuffle=(split == "train"),
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        print(f"  {split}: {len(combined)} samples")
    return loaders["train"], loaders["val"], loaders["test"]


# ── Model builders ─────────────────────────────────────────────────────────────

def build_model_for_training(cfg: dict, production_bytes: bytes, device) -> nn.Module:
    backbone = cfg["backbone"]
    if backbone == "resnet50":
        ckpt = torch.load(io.BytesIO(production_bytes), map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        saved_cfg = ckpt.get("exp_cfg", {})
        model = GazeClassifier(
            backbone=resnet50(),
            num_classes=NUM_CLASSES,
            head_dense_units=saved_cfg.get("head_dense_units", []),
            dropout=cfg.get("dropout", 0.1),
            batch_norm_in_head=cfg.get("batch_norm_in_head", True),
        )
        model.load_state_dict(state_dict, strict=False)
    elif backbone == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    elif backbone == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model.to(device)


def build_model_for_eval(cfg: dict, device) -> nn.Module:
    backbone = cfg["backbone"]
    if backbone == "resnet50":
        model = GazeClassifier(
            backbone=resnet50(),
            num_classes=NUM_CLASSES,
            head_dense_units=cfg.get("head_dense_units", []),
            dropout=cfg.get("dropout", 0.1),
            batch_norm_in_head=cfg.get("batch_norm_in_head", True),
        )
    elif backbone == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    elif backbone == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model.to(device)


# ── Freeze / unfreeze ──────────────────────────────────────────────────────────

def setup_phase1(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def setup_phase2(model: nn.Module, cfg: dict) -> None:
    backbone_name = cfg["backbone"]
    unfreeze_from = cfg["unfreeze_from"]
    if backbone_name == "resnet50":
        layer_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "avgpool"]
        unfreeze_idx = layer_order.index(unfreeze_from)
        for name, module in model.backbone.named_children():
            if name in layer_order and layer_order.index(name) >= unfreeze_idx:
                for param in module.parameters():
                    param.requires_grad = True
    else:
        prefix, idx_str = unfreeze_from.split(".")
        unfreeze_idx = int(idx_str)
        for i, block in enumerate(getattr(model, prefix)):
            if i >= unfreeze_idx:
                for param in block.parameters():
                    param.requires_grad = True


# ── Training loop helpers ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
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


class EarlyStopping:
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
        if self.best_state:
            model.load_state_dict(self.best_state)


def run_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
              num_epochs, device, phase_name, patience, logger=None, epoch_offset=0):
    stopper = EarlyStopping(patience)
    history = []
    for epoch in range(num_epochs):
        g = epoch_offset + epoch + 1
        _, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_loader(model, val_loader, criterion, device)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(va_loss)
        elif scheduler:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"  [{phase_name}] {epoch+1}/{num_epochs} | train {tr_acc:.4f} | val {va_acc:.4f} | lr {lr:.2e}")
        if logger:
            logger.report_scalar("accuracy", f"{phase_name}_train", tr_acc, g)
            logger.report_scalar("accuracy", f"{phase_name}_val", va_acc, g)
        history.append({"val_accuracy": va_acc})
        stopper(va_acc, model)
        if stopper.should_stop:
            print(f"  Early stopping at epoch {epoch + 1}")
            break
    stopper.restore_best(model)
    return history


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def eval_loader(model, loader, criterion, device):
    """Simple eval — returns (loss, acc). No FPS or confusion matrix."""
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            out = model(images)
            loss_sum += criterion(out, targets).item() * images.size(0)
            correct += out.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return (loss_sum / total, correct / total) if total else (0.0, 0.0)


def eval_with_metrics(model, loader, criterion, device, name: str = "", logger=None):
    """
    Full eval pass that also measures inference FPS and builds a confusion matrix.
    Returns (loss, acc, fps, confusion_matrix_list).
    """
    model.eval()

    # Warm-up (stabilises GPU timing)
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= 3:
                break
            model(imgs.to(device))
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    loss_sum = correct = total = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            out = model(images)
            loss_sum += criterion(out, targets).item() * images.size(0)
            preds = out.argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    loss = loss_sum / total if total else 0.0
    acc  = correct  / total if total else 0.0
    fps  = total / elapsed  if elapsed > 0 else 0.0

    # Confusion matrix (rows = true, cols = predicted)
    cm = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for p, t in zip(all_preds, all_targets):
        cm[t][p] += 1

    print(f"  {name}: acc={acc:.4f} | fps={fps:.1f} img/s")

    if logger and name:
        logger.report_single_value(f"fps/{name}", fps)
        logger.report_confusion_matrix(
            title="Confusion Matrix",
            series=name,
            iteration=0,
            matrix=cm,
            xlabels=LABELS,
            ylabels=LABELS,
        )

    return loss, acc, fps, cm


# ── High-level training entry point ───────────────────────────────────────────

def train_single_model(config_name: str, user_data_dir: str, logger=None) -> dict:
    cfg = MODEL_CONFIGS[config_name]

    seed_everything()
    device = get_runtime_device()
    print(f"\n=== Training {config_name} on {get_runtime_device_str()} ===")

    production_bytes = download_backbone()
    train_loader, val_loader, _ = build_loaders(user_data_dir)

    candidates_dir = os.path.join(PIPELINE_TMP_DIR, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    model = build_model_for_training(cfg, production_bytes, device)
    patience = cfg.get("early_stopping_patience", 10)
    history = []

    print("  Phase 1: classifier head only")
    setup_phase1(model)
    opt1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["phase1_lr"])
    sched1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, mode="min", factor=0.5, patience=2)
    history += run_phase(model, train_loader, val_loader, criterion, opt1, sched1,
                         cfg["phase1_epochs"], device, "phase1", patience, logger)

    if cfg.get("phase2_epochs", 0) > 0:
        print(f"  Phase 2: unfreeze from {cfg['unfreeze_from']}")
        setup_phase2(model, cfg)
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and ("backbone" in n or "features" in n)]
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "backbone" not in n and "features" not in n]
        opt2 = optim.Adam([
            {"params": backbone_params, "lr": cfg["phase2_lr"] * 0.1},
            {"params": head_params,     "lr": cfg["phase2_lr"]},
        ])
        sched2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=3)
        history += run_phase(model, train_loader, val_loader, criterion, opt2, sched2,
                             cfg["phase2_epochs"], device, "phase2", patience, logger,
                             epoch_offset=len(history))

    best_val = max(h["val_accuracy"] for h in history) if history else 0.0
    ckpt_path = os.path.join(candidates_dir, f"{config_name}.pth")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "config_name": config_name}, ckpt_path)
    print(f"  {config_name}: best val acc = {best_val:.4f}")
    if logger:
        logger.report_single_value(f"train_val_acc/{config_name}", best_val)

    return {"config_name": config_name, "checkpoint_path": ckpt_path, "train_val_acc": best_val}
