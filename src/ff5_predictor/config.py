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
            "IWN",
            "IWO",
            "IWD",
            "IWF",
            "IJR",
            "IJH",
            "RSP",
            "SPHQ",
            "VNQ",
            "EFA",
            "EEM",
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
        "include_lagged_targets": False,
        "lags": [],
        "rolling_windows": [],
        "include_rolling_mean": False,
        "include_rolling_volatility": False,
        "include_rolling_min_max": False,
        "include_rf_lags": False,
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
        "recursive_factor_lags": False,
        "release_gap_backtest_days": [1, 2, 3, 5, 10],
    },
    "output": {
        "force_overwrite": False,
    },
    "nowcast": {
        "output_dir": "data/nowcasts",
        "run_name": "latest",
        "models": ["ridge"],
        "primary_model": "ridge",
        "estimate_unreleased_only": True,
        "save_feature_snapshot": True,
        "save_model_artifact": True,
        "save_feature_attributions": True,
        "min_train_rows": 1000,
        "train_window_rows": 2520,
        "backtest_step_rows": 21,
    },
    "attribution": {
        "top_n": 20,
    },
    "backtest": {
        "n_jobs": 1,
        "backend": "loky",
        "verbose": 0,
    },
    "feature_extraction": {
        "enabled": False,
        "method": "none",
        "apply_to_models": ["ridge"],
        "output_prefix": "fx",
        "save_transformer_artifact": False,
        "keep_original_features": False,
        "group_pca": {
            "scale_before_pca": True,
            "groups": {
                "market_returns": {"patterns": ["*_ret_1d", "*_log_ret_1d"], "n_components": 8},
                "ohlc_intraday": {"patterns": ["*_oc_ret", "*_hl_range", "*_gap", "*_hl_range_mean_*"], "n_components": 8},
                "rolling_returns": {"patterns": ["*_ret_*d"], "n_components": 10},
                "rolling_volatility": {"patterns": ["*_vol_*d"], "n_components": 10},
                "drawdown": {"patterns": ["*_drawdown_*d"], "n_components": 5},
                "proxy_size": {"patterns": ["proxy_size_*"], "n_components": 4},
                "proxy_value": {"patterns": ["proxy_value_*", "proxy_growth_*"], "n_components": 4},
                "proxy_sector": {"patterns": ["proxy_sector_*"], "n_components": 6},
                "proxy_risk": {"patterns": ["proxy_credit_*", "proxy_vix_*", "proxy_tlt_*"], "n_components": 4},
                "proxy_global": {"patterns": ["proxy_global_*"], "n_components": 3},
                "proxy_quality": {"patterns": ["proxy_quality_*"], "n_components": 2},
                "proxy_realestate": {"patterns": ["proxy_realestate_*"], "n_components": 2},
                "other": {"patterns": ["*"], "n_components": 10},
            },
        },
        "pls": {"n_components": 20, "scale_features": True, "scale_targets": False},
        "per_target_pls": {"n_components": 10, "scale_features": True, "scale_targets": False},
        "clustered": {
            "correlation_threshold": 0.92,
            "max_features_for_clustering": 2000,
            "min_cluster_size": 2,
            "singleton_policy": "keep",
            "representative": "mean",
            "scale_before_clustering": True,
        },
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
        "elasticnet": {
            "alpha": 0.001,
            "alpha_grid": [0.0003, 0.001, 0.003, 0.01],
            "l1_ratio": 0.05,
            "max_iter": 50000,
            "tol": 0.0001,
            "tune_alpha": True,
            "validation_window_rows": 252,
            "scale_features": True,
        },
        "per_factor_elasticnet": {
            "alpha": 0.001,
            "alpha_grid": [0.0003, 0.001, 0.003, 0.01],
            "l1_ratio": 0.05,
            "max_iter": 50000,
            "tol": 0.0001,
            "tune_alpha": True,
            "validation_window_rows": 252,
            "scale_features": True,
        },
        "ewma": {"spans": [5, 21, 63], "default_span": 21},
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
    for section in ("data", "features", "prediction", "output", "nowcast"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    tickers = config["data"].get("tickers")
    if not tickers or not all(isinstance(t, str) and t.strip() for t in tickers):
        raise ValueError("data.tickers must contain at least one ticker")

    if int(config["prediction"].get("horizon", 1)) < 1:
        raise ValueError("prediction.horizon must be at least 1")

    targets = config["prediction"].get("target_columns")
    if not targets:
        raise ValueError("prediction.target_columns must not be empty")


def path_from_config(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config
    for key in keys:
        value = value[key]
    return Path(value)
