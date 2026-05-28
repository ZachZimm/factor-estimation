from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


def infer_tickers(market_df: pd.DataFrame) -> list[str]:
    tickers: set[str] = set()
    for column in market_df.columns:
        for field in OHLCV_FIELDS:
            suffix = f"_{field}"
            if column.endswith(suffix):
                tickers.add(column[: -len(suffix)])
    return sorted(tickers)


def build_market_features(market_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    feature_series: dict[str, pd.Series] = {}
    windows = [int(window) for window in config["features"].get("lookback_windows", [])]
    include_returns = bool(config["features"].get("include_returns", True))
    include_log_returns = bool(config["features"].get("include_log_returns", True))
    include_rolling_volatility = bool(
        config["features"].get("include_rolling_volatility", True)
    )
    include_rolling_volume = bool(
        config["features"].get("include_rolling_volume_features", True)
    )
    include_ohlc = bool(config["features"].get("include_ohlc_features", True))
    include_drawdown = bool(config["features"].get("include_drawdown", True))

    for ticker in infer_tickers(market_df):
        open_ = market_df[f"{ticker}_open"]
        high = market_df[f"{ticker}_high"]
        low = market_df[f"{ticker}_low"]
        close = market_df[f"{ticker}_close"]
        volume = market_df[f"{ticker}_volume"]

        ret_1d = close.pct_change()
        hl_range = (high - low) / close

        if include_returns:
            feature_series[f"{ticker}_ret_1d"] = ret_1d
        if include_log_returns:
            feature_series[f"{ticker}_log_ret_1d"] = np.log(close).diff()
        if include_ohlc:
            feature_series[f"{ticker}_oc_ret"] = close / open_ - 1
            feature_series[f"{ticker}_hl_range"] = hl_range
            feature_series[f"{ticker}_gap"] = open_ / close.shift(1) - 1

        for window in windows:
            if include_returns:
                feature_series[f"{ticker}_ret_{window}d"] = close / close.shift(window) - 1
            if include_rolling_volatility:
                feature_series[f"{ticker}_vol_{window}d"] = ret_1d.rolling(window).std(ddof=0)
            if include_rolling_volume:
                volume_mean = volume.rolling(window).mean()
                feature_series[f"{ticker}_volume_mean_{window}d"] = volume_mean
                if window >= 2:
                    volume_std = volume.rolling(window).std(ddof=0)
                    feature_series[f"{ticker}_volume_z_{window}d"] = (
                        (volume - volume_mean) / volume_std.replace(0, np.nan)
                    )
            if include_ohlc:
                feature_series[f"{ticker}_hl_range_mean_{window}d"] = hl_range.rolling(window).mean()
            if include_drawdown:
                feature_series[f"{ticker}_drawdown_{window}d"] = close / close.rolling(window).max() - 1

        if not bool(config["features"].get("drop_raw_ohlcv", False)):
            for field in OHLCV_FIELDS:
                feature_series[f"{ticker}_{field}"] = market_df[f"{ticker}_{field}"]

    if not feature_series:
        return pd.DataFrame(index=market_df.index)

    features = pd.DataFrame(feature_series, index=market_df.index)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna(axis=1, how="all")
