from __future__ import annotations

from typing import Any

from ff5_predictor.torch_models.ft_transformer import SequenceFTTransformerRegressor
from ff5_predictor.torch_models.tcn import TCNRegressor
from ff5_predictor.torch_models.tft import TFTStyleRegressor


TORCH_MODEL_TYPES = {"tft", "torch_tft", "tcn", "torch_tcn", "ft_transformer", "torch_ft_transformer"}


def normalize_torch_model_type(model_type: str) -> str:
    aliases = {
        "torch_tft": "tft",
        "torch_tcn": "tcn",
        "torch_ft_transformer": "ft_transformer",
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
    if normalized == "tft":
        return TFTStyleRegressor(
            n_features=n_features,
            n_targets=n_targets,
            hidden_size=int(cfg.get("hidden_size", torch_cfg.get("hidden_size", 128))),
            n_heads=int(cfg.get("n_heads", 4)),
            lstm_layers=int(cfg.get("lstm_layers", 1)),
            dropout=dropout,
        )
    if normalized == "tcn":
        return TCNRegressor(
            n_features=n_features,
            n_targets=n_targets,
            channels=[int(v) for v in cfg.get("channels", [64, 64, 128])],
            kernel_size=int(cfg.get("kernel_size", 3)),
            dropout=dropout,
        )
    if normalized == "ft_transformer":
        return SequenceFTTransformerRegressor(
            n_features=n_features,
            n_targets=n_targets,
            d_model=int(cfg.get("d_model", torch_cfg.get("hidden_size", 128))),
            n_heads=int(cfg.get("n_heads", 4)),
            num_layers=int(cfg.get("num_layers", 2)),
            dim_feedforward=int(cfg.get("dim_feedforward", 256)),
            dropout=dropout,
        )
    raise ValueError(f"Unsupported torch model_type: {model_type}")
