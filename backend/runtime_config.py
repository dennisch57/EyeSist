import os
from functools import lru_cache

import torch


DEVICE_ENV_VAR = "EYESIST_DEVICE"


@lru_cache(maxsize=1)
def get_runtime_device() -> torch.device:
    requested = os.getenv(DEVICE_ENV_VAR, "auto").strip().lower()

    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unsupported {DEVICE_ENV_VAR} value '{requested}'. "
            "Expected one of: auto, cpu, cuda, mps."
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("EYESIST_DEVICE=cuda was requested, but CUDA is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("EYESIST_DEVICE=mps was requested, but MPS is not available.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


@lru_cache(maxsize=1)
def get_runtime_device_str() -> str:
    return str(get_runtime_device())
