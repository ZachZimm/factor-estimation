from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "task": {
        "mode": "nowcast",
        "estimate_unreleased_only": True,
    },
    "data": {
        "start_date": "2000-01-01",
        "end_date": None,
        "ff5_frequency": "daily",
        "tickers": [
            "SPY",
            "VTI",
            "IWM",
            "QQQ",
            "IVE",
            "IVW",
            "VBR",
            "VBK",
            "IJS",
            "IJT",
            "XLF",
            "XLK",
            "XLI",
            "XLY",
            "XLP",
            "XLV",
            "XLE",
            "XLU",
            "TLT",
            "IEF",
            "SHY",
            "HYG",
            "LQD",
            "GLD",
            "UUP",
            "^VIX",
        ],
        "cache_dir": "data/cache",
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "force_refresh": False,
    },
    "features": {
        "lookback_windows": [1, 2, 5, 10, 21, 63, 126],
        "include_returns": True,
        "include_log_returns": True,
        "include_rolling_volatility": True,
        "include_rolling_volume_features": False,
        "include_ohlc_features": True,
        "include_drawdown": True,
        "drop_raw_ohlcv": True,
    },
    "target_features": {
        "include_lagged_targets": True,
        "lags": [1, 2, 5, 10, 21],
        "rolling_windows": [5, 21, 63],
        "include_rolling_mean": True,
        "include_rolling_volatility": True,
        "include_rolling_min_max": False,
        "include_rf_lags": True,
    },
    "training": {
        "train_window_days": 1260,
        "min_train_rows": 1000,
        "step_size": 1,
        "model_type": "ridge",
        "scale_features": True,
        "refit_each_step": True,
        "save_models": False,
    },
    "prediction": {
        "horizon": 1,
        "target_mode": "daily",
        "cumulative_horizon_rows": 1,
        "target_columns": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
        "include_rf_as_feature": False,
        "predict_rf": False,
    },
    "availability": {
        "market_data_lag_rows": 0,
        "official_factor_lag_rows": 1,
        "recursive_factor_lags": True,
        "release_gap_backtest_days": [1, 2, 3, 5, 10],
    },
    "output": {
        "predictions_path": "data/predictions/ff5_predictions.csv",
        "metrics_path": "data/predictions/metrics.json",
        "models_dir": "data/models",
        "force_overwrite": False,
    },
    "nowcast": {
        "output_dir": "data/nowcasts",
        "run_name": "daily_ff5_nowcast_v1",
        "models": ["rolling_mean", "ewma", "ridge", "tft"],
        "primary_model": "ridge",
        "estimate_unreleased_only": True,
        "save_feature_snapshot": True,
        "save_model_artifact": True,
        "min_train_rows": 1000,
        "train_window_rows": 2520,
        "backtest_step_rows": 21,
    },
    "experiments": {
        "output_dir": "data/experiments",
        "run_name": None,
        "random_seed": 42,
        "models": [
            "rolling_mean",
            "ewma",
            "elasticnet",
            "ridge",
            "patchtst",
            "tft",
        ],
    },
    "walk_forward": {
        "checkpoint_frequency": "M",
        "train_window_rows": 1260,
        "min_train_rows": 1000,
        "min_fit_rows": 1000,
        "validation_window_rows": 252,
        "require_validation": False,
        "predict_between_checkpoints": True,
        "refit_on_each_checkpoint": True,
    },
    "sequence": {
        "lookback_rows": 63,
        "batch_size": 256,
        "num_workers": 0,
        "drop_incomplete_sequences": True,
    },
    "torch": {
        "device": "auto",
        "max_epochs": 50,
        "patience": 8,
        "learning_rate": 0.0005,
        "weight_decay": 0.001,
        "gradient_clip_norm": 1.0,
        "mixed_precision": False,
        "hidden_size": 128,
        "dropout": 0.15,
        "standardize_targets": True,
        "save_checkpoints": False,
        "log_device": True,
        "restore_best_checkpoint": True,
    },
    "models": {
        "ridge": {
            "alpha": 10.0,
            "alpha_grid": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
            "tune_alpha": True,
            "validation_window_rows": 252,
            "scale_features": True,
        },
        "elasticnet": {"alpha": 0.001, "l1_ratio": 0.2, "max_iter": 50000, "tol": 0.0001},
        "ewma": {"spans": [5, 21, 63], "default_span": 21},
        "mlp_window": {"hidden_sizes": [512, 256, 128], "activation": "gelu"},
        "tcn": {"channels": [64, 64, 128, 128], "kernel_size": 3, "dropout": 0.15},
        "patchtst": {
            "enabled": False,
            "hidden": True,
            "patch_len": 8,
            "stride": 4,
            "d_model": 64,
            "n_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.25,
        },
        "tft": {"enabled": True, "hidden_size": 64, "n_heads": 4, "lstm_layers": 1, "dropout": 0.25},
    },
    "proxy_features": {
        "enabled": True,
        "include_relative_returns": True,
        "include_factor_mimic_spreads": True,
        "include_risk_proxies": True,
        "rolling_windows": [5, 21, 63],
    },
    "fundamentals": {
        "enabled": False,
    },
    "regimes": {
        "enabled": True,
        "volatility_source": "SPY_vol_21d",
        "high_vol_quantile": 0.8,
        "low_vol_quantile": 0.2,
        "market_return_source": "SPY_ret_21d",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    config = deep_merge(DEFAULT_CONFIG, loaded)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "features", "training", "prediction", "output"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    tickers = config["data"].get("tickers")
    if not tickers or not all(isinstance(t, str) and t.strip() for t in tickers):
        raise ValueError("data.tickers must contain at least one ticker")

    if int(config["prediction"].get("horizon", 1)) < 1:
        raise ValueError("prediction.horizon must be at least 1")

    if int(config["training"].get("train_window_days", 0)) < 1:
        raise ValueError("training.train_window_days must be positive")

    targets = config["prediction"].get("target_columns")
    if not targets:
        raise ValueError("prediction.target_columns must not be empty")


def path_from_config(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config
    for key in keys:
        value = value[key]
    return Path(value)
