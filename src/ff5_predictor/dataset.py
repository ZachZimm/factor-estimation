from __future__ import annotations

from typing import Any

import pandas as pd

from ff5_predictor.features import build_market_features
from ff5_predictor.io import normalize_datetime_index
from ff5_predictor.target_features import build_lagged_target_features, is_target_feature_column


def build_modeling_dataset(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    ff5 = normalize_datetime_index(ff5_df)
    market = normalize_datetime_index(market_df)
    target_columns = list(config["prediction"]["target_columns"])
    missing_targets = [col for col in target_columns if col not in ff5.columns]
    if missing_targets:
        raise ValueError(f"FF5 data missing target columns: {missing_targets}")

    features = build_market_features(market, config)
    horizon = int(config["prediction"].get("horizon", 1))
    # No-leakage boundary: row t receives feature values from t-horizon, so
    # prediction for FF5 date t cannot see same-day or future market data.
    shifted_features = features.shift(horizon)
    target_features = build_lagged_target_features(ff5, config)
    if not target_features.empty:
        shifted_features = shifted_features.join(target_features, how="inner")

    if bool(config["prediction"].get("include_rf_as_feature", False)):
        shifted_features["RF_lagged"] = ff5["RF"].shift(horizon)

    modeling_df = shifted_features.join(ff5[target_columns], how="inner")
    modeling_df = modeling_df.dropna(subset=list(shifted_features.columns) + target_columns)
    modeling_df = normalize_datetime_index(modeling_df)
    return modeling_df


def split_feature_target_columns(
    df: pd.DataFrame,
    target_columns: list[str],
) -> tuple[list[str], list[str]]:
    missing = [col for col in target_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing target columns: {missing}")
    feature_columns = [col for col in df.columns if col not in target_columns]
    return feature_columns, target_columns


def get_feature_groups(df: pd.DataFrame, target_columns: list[str]) -> dict[str, list[str]]:
    feature_columns = [col for col in df.columns if col not in target_columns]
    factor_columns = [col for col in feature_columns if is_target_feature_column(col)]
    market_columns = [col for col in feature_columns if col not in factor_columns]
    return {
        "all": feature_columns,
        "market": market_columns,
        "factor": factor_columns,
        "target": target_columns,
    }
