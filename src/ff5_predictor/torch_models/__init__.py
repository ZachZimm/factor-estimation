from __future__ import annotations

from typing import Any

from ff5_predictor.torch_models.mlp_window import WindowMLP
from ff5_predictor.torch_models.patchtst import PatchTSTRegressor
from ff5_predictor.torch_models.tcn import TCNRegressor
from ff5_predictor.torch_models.tft import TFTStyleRegressor


TORCH_MODEL_TYPES = {"mlp_window", "tcn", "patchtst", "tft", "torch_mlp", "torch_tcn", "torch_patchtst", "torch_tft"}


def normalize_torch_model_type(model_type: str) -> str:
    aliases = {
        "torch_mlp": "mlp_window",
        "torch_tcn": "tcn",
        "torch_patchtst": "patchtst",
        "torch_tft": "tft",
    }
    return aliases.get(model_type, model_type)


def make_torch_model(
    model_type: str,
    lookback_rows: int,
    n_features: int,
    n_targets: int,
    config: dict[str, Any],
):
    normalized = normalize_torch_model_type(model_type)
    cfg = config.get("models", {}).get(normalized, {})
    torch_cfg = config.get("torch", {})
    dropout = float(cfg.get("dropout", torch_cfg.get("dropout", 0.15)))
    if normalized == "mlp_window":
        return WindowMLP(
            lookback_rows=lookback_rows,
            n_features=n_features,
            n_targets=n_targets,
            hidden_sizes=list(cfg.get("hidden_sizes", [512, 256, 128])),
            dropout=dropout,
            activation=str(cfg.get("activation", "gelu")),
        )
    if normalized == "tcn":
        return TCNRegressor(
            n_features=n_features,
            n_targets=n_targets,
            channels=list(cfg.get("channels", [64, 64, 128, 128])),
            kernel_size=int(cfg.get("kernel_size", 3)),
            dropout=dropout,
        )
    if normalized == "patchtst":
        return PatchTSTRegressor(
            lookback_rows=lookback_rows,
            n_features=n_features,
            n_targets=n_targets,
            patch_len=int(cfg.get("patch_len", 8)),
            stride=int(cfg.get("stride", 4)),
            d_model=int(cfg.get("d_model", 128)),
            n_heads=int(cfg.get("n_heads", 4)),
            num_layers=int(cfg.get("num_layers", 3)),
            dim_feedforward=int(cfg.get("dim_feedforward", 256)),
            dropout=dropout,
        )
    if normalized == "tft":
        return TFTStyleRegressor(
            n_features=n_features,
            n_targets=n_targets,
            hidden_size=int(cfg.get("hidden_size", torch_cfg.get("hidden_size", 128))),
            n_heads=int(cfg.get("n_heads", 4)),
            lstm_layers=int(cfg.get("lstm_layers", 1)),
            dropout=dropout,
        )
    raise ValueError(f"Unsupported torch model_type: {model_type}")
