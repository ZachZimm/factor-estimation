from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ff5_predictor.io import normalize_datetime_index


def build_lagged_target_features(ff5_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    target_config = config.get("target_features", {})
    if not target_config.get("include_lagged_targets", False):
        return pd.DataFrame(index=normalize_datetime_index(ff5_df).index)

    ff5 = normalize_datetime_index(ff5_df)
    targets = list(config.get("prediction", {}).get("target_columns", []))
    if not targets:
        targets = [col for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if col in ff5.columns]
    horizon = int(config.get("prediction", {}).get("horizon", 1))
    lags = [int(lag) for lag in target_config.get("lags", [])]
    windows = [int(window) for window in target_config.get("rolling_windows", [])]

    features = pd.DataFrame(index=ff5.index)
    shifted_targets = ff5[targets].shift(horizon)

    for target in targets:
        source = shifted_targets[target]
        for lag in lags:
            effective_lag = max(lag, horizon)
            features[f"{target}_lag_{lag}"] = ff5[target].shift(effective_lag)
        for window in windows:
            if target_config.get("include_rolling_mean", True):
                features[f"{target}_mean_{window}"] = source.rolling(window).mean()
            if target_config.get("include_rolling_volatility", True):
                features[f"{target}_vol_{window}"] = source.rolling(window).std(ddof=0)
            if target_config.get("include_rolling_min_max", True):
                features[f"{target}_min_{window}"] = source.rolling(window).min()
                features[f"{target}_max_{window}"] = source.rolling(window).max()

    if target_config.get("include_rf_lags", False) and "RF" in ff5.columns:
        for lag in lags:
            features[f"RF_lag_{lag}"] = ff5["RF"].shift(max(lag, horizon))

    return features.replace([np.inf, -np.inf], np.nan)


def is_target_feature_column(column: str) -> bool:
    markers = ("_lag_", "_mean_", "_vol_", "_min_", "_max_")
    return column == "RF_lagged" or any(marker in column for marker in markers)
