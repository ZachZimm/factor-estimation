from __future__ import annotations

from typing import Any

from ff5_predictor.torch_models.tft import TFTStyleRegressor


TORCH_MODEL_TYPES = {"tft", "torch_tft"}


def normalize_torch_model_type(model_type: str) -> str:
    aliases = {
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
