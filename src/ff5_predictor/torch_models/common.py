from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


class WindowStandardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "WindowStandardizer":
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("WindowStandardizer is not fitted")
        return ((X - self.mean_) / self.std_).astype(np.float32)


class TargetStandardizer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "TargetStandardizer":
        self.mean_ = y.mean(axis=0)
        self.std_ = y.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return y.astype(np.float32)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("TargetStandardizer is not fitted")
        return ((y - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return y
        if self.mean_ is None or self.std_ is None:
            raise ValueError("TargetStandardizer is not fitted")
        return y * self.std_ + self.mean_


@dataclass
class EarlyStopping:
    patience: int
    best_score: float = float("inf")
    epochs_without_improvement: int = 0

    def step(self, score: float) -> bool:
        if score < self.best_score:
            self.best_score = score
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def select_device(config: dict) -> "object":
    import torch

    requested = str(config.get("torch", {}).get("device", "auto")).lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch.device is configured as 'cuda', but CUDA is not available")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def activation_layer(name: str):
    import torch

    normalized = name.lower()
    if normalized == "relu":
        return torch.nn.ReLU()
    if normalized == "gelu":
        return torch.nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")
