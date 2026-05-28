from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ff5_predictor.features import build_market_features
from ff5_predictor.fundamentals import build_fundamentals_features
from ff5_predictor.io import normalize_datetime_index
from ff5_predictor.proxy_features import build_proxy_features


@dataclass(frozen=True)
class NowcastFeatureResult:
    features: pd.DataFrame
    feature_columns: list[str]
    market_feature_columns: list[str]
    proxy_feature_columns: list[str]
    factor_feature_columns: list[str]
    metadata: dict[str, Any]


def build_nowcast_features(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: dict[str, Any],
    official_cutoff_date: pd.Timestamp | None = None,
    recursive_predictions: pd.DataFrame | None = None,
) -> NowcastFeatureResult:
    ff5 = normalize_datetime_index(ff5_df)
    raw_market = normalize_datetime_index(market_df)
    market = _drop_sparse_market_rows(raw_market, config)
    market_features = build_market_features(market, config)
    market_lag = int(config.get("availability", {}).get("market_data_lag_rows", 0))
    if market_lag:
        market_features = market_features.shift(market_lag)

    proxy_features = build_proxy_features(market_features, config)
    factor_features = _build_factor_features_for_nowcast(
        ff5,
        pd.DatetimeIndex(market_features.index),
        config,
        official_cutoff_date=official_cutoff_date,
        recursive_predictions=recursive_predictions,
    )
    fundamentals = build_fundamentals_features(config, pd.DatetimeIndex(market_features.index))

    frames = [market_features, proxy_features, factor_features, fundamentals.features]
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if non_empty_frames:
        features = pd.concat(non_empty_frames, axis=1)
        features = features.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    else:
        features = pd.DataFrame(index=market.index)
    features = normalize_datetime_index(features)

    market_cols = list(market_features.columns)
    proxy_cols = list(proxy_features.columns)
    factor_cols = list(factor_features.columns)
    return NowcastFeatureResult(
        features=features,
        feature_columns=list(features.columns),
        market_feature_columns=[col for col in market_cols if col in features.columns],
        proxy_feature_columns=[col for col in proxy_cols if col in features.columns],
        factor_feature_columns=[col for col in factor_cols if col in features.columns],
        metadata={
            "market_data_lag_rows": market_lag,
            "official_factor_lag_rows": int(config.get("availability", {}).get("official_factor_lag_rows", 1)),
            "official_cutoff_date": None if official_cutoff_date is None else str(pd.Timestamp(official_cutoff_date).date()),
            "recursive_prediction_rows": 0 if recursive_predictions is None else int(len(recursive_predictions)),
            "dropped_sparse_market_rows": int(len(raw_market) - len(market)),
            "fundamentals": fundamentals.metadata,
        },
    )


def _drop_sparse_market_rows(market: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Remove partial holiday/current rows before rolling feature construction.

    yfinance can occasionally return a row where only one instrument, commonly
    ^VIX, has values while ETF markets are closed. Keeping that row poisons
    rolling features for subsequent valid trading days.
    """
    if market.empty:
        return market
    min_non_null_fraction = float(config.get("data", {}).get("min_market_row_non_null_fraction", 0.95))
    threshold = max(1, int(np.ceil(market.shape[1] * min_non_null_fraction)))
    return market.loc[market.notna().sum(axis=1) >= threshold]


def _build_factor_features_for_nowcast(
    ff5: pd.DataFrame,
    feature_index: pd.DatetimeIndex,
    config: dict[str, Any],
    official_cutoff_date: pd.Timestamp | None,
    recursive_predictions: pd.DataFrame | None,
) -> pd.DataFrame:
    target_config = config.get("target_features", {})
    if not target_config.get("include_lagged_targets", False):
        return pd.DataFrame(index=feature_index)

    target_columns = list(config.get("prediction", {}).get("target_columns", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]))
    lag_rows = int(config.get("availability", {}).get("official_factor_lag_rows", 1))
    cutoff = pd.Timestamp(official_cutoff_date) if official_cutoff_date is not None else None
    source = ff5.copy()
    if cutoff is not None:
        source = source.loc[:cutoff]
    source = source[[col for col in target_columns + ["RF"] if col in source.columns]]
    predicted = _predictions_to_factor_frame(recursive_predictions, target_columns)
    if predicted is not None and not predicted.empty:
        if cutoff is not None:
            predicted = predicted.loc[predicted.index > cutoff]
        source = pd.concat([source, predicted], axis=0)
    source = source[~source.index.duplicated(keep="last")].sort_index()
    if "RF" in source.columns:
        source["RF"] = source["RF"].ffill()

    full_index = pd.DatetimeIndex(feature_index.union(source.index)).sort_values()
    source = source.reindex(full_index)
    if "RF" in source.columns:
        source["RF"] = source["RF"].ffill()
    features: dict[str, pd.Series] = {}
    lags = [int(lag) for lag in target_config.get("lags", [])]
    windows = [int(window) for window in target_config.get("rolling_windows", [])]
    shifted_targets = source[target_columns].shift(lag_rows)

    for target in target_columns:
        if target not in source.columns:
            continue
        for lag in lags:
            features[f"{target}_lag_{lag}"] = source[target].shift(max(lag, lag_rows))
        base = shifted_targets[target]
        for window in windows:
            if target_config.get("include_rolling_mean", True):
                features[f"{target}_mean_{window}"] = base.rolling(window).mean()
            if target_config.get("include_rolling_volatility", True):
                features[f"{target}_vol_{window}"] = base.rolling(window).std(ddof=0)
            if target_config.get("include_rolling_min_max", False):
                features[f"{target}_min_{window}"] = base.rolling(window).min()
                features[f"{target}_max_{window}"] = base.rolling(window).max()

    if target_config.get("include_rf_lags", False) and "RF" in source.columns:
        for lag in lags:
            features[f"RF_lag_{lag}"] = source["RF"].shift(max(lag, lag_rows))

    if not features:
        return pd.DataFrame(index=feature_index)
    result = pd.DataFrame(features, index=full_index).reindex(feature_index)
    return result.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")


def _predictions_to_factor_frame(
    recursive_predictions: pd.DataFrame | None,
    target_columns: list[str],
) -> pd.DataFrame | None:
    if recursive_predictions is None or recursive_predictions.empty:
        return None
    predictions = recursive_predictions.copy()
    if "date" in predictions.columns:
        predictions.index = pd.to_datetime(predictions["date"])
    predictions = normalize_datetime_index(predictions)
    data: dict[str, pd.Series] = {}
    for target in target_columns:
        if target in predictions.columns:
            data[target] = predictions[target]
        elif f"pred_{target}" in predictions.columns:
            data[target] = predictions[f"pred_{target}"]
    if not data:
        return None
    return pd.DataFrame(data, index=predictions.index)
