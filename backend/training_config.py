import random

import numpy as np
import torch


DATA_DIR = "./calibration_data"
TRAIN_DIR = f"{DATA_DIR}/train"
VAL_DIR = f"{DATA_DIR}/val"
TEST_DIR = f"{DATA_DIR}/test"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 25221627
NUM_WORKERS = 4

LABELS = ['closed', 'down', 'left', 'right', 'straight', 'up']
NUM_CLASSES = len(LABELS)

# Dataset manifest split targets (user sessions only; original dataset split is fixed)
SPLIT_TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Volume-based retrain trigger thresholds
RETRAIN_NEW_SAMPLES_THRESHOLD = 5000   # trigger if 5000+ new train samples since last retrain

RETRAIN_EXPERIMENT_CONFIG = {
    "name": "ethxgaze_head",
    "phase1_epochs": 15,
    "phase2_epochs": 80,
    "phase1_lr": 1e-3,
    "phase2_lr": 1e-4,
    "dropout": 0.1,
    "head_dense_units": [],
    "batch_norm_in_head": True,
    "unfreeze_from_layer": "layer3",
    "unfreeze_batchnorm": False,
    "early_stopping_patience": 10,
}


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
