from __future__ import annotations

import pandas as pd

from ff5_predictor.dataset import build_modeling_dataset


def _config() -> dict:
    return {
        "features": {
            "lookback_windows": [],
            "include_returns": False,
            "include_log_returns": False,
            "include_rolling_volatility": False,
            "include_rolling_volume_features": False,
            "include_ohlc_features": False,
            "include_drawdown": False,
            "drop_raw_ohlcv": False,
        },
        "prediction": {
            "horizon": 1,
            "target_columns": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
            "include_rf_as_feature": False,
        },
    }


def test_feature_shift_no_leakage() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    market = pd.DataFrame(
        {
            "SPY_open": [10, 20, 30, 40],
            "SPY_high": [11, 21, 31, 41],
            "SPY_low": [9, 19, 29, 39],
            "SPY_close": [100, 200, 300, 400],
            "SPY_volume": [1000, 2000, 3000, 4000],
        },
        index=dates,
    )
    ff5 = pd.DataFrame(
        {
            "Mkt-RF": [0.01, 0.02, 0.03, 0.04],
            "SMB": [0.01, 0.02, 0.03, 0.04],
            "HML": [0.01, 0.02, 0.03, 0.04],
            "RMW": [0.01, 0.02, 0.03, 0.04],
            "CMA": [0.01, 0.02, 0.03, 0.04],
            "RF": [0.0, 0.0, 0.0, 0.0],
        },
        index=dates,
    )

    dataset = build_modeling_dataset(ff5, market, _config())

    assert dataset.loc[pd.Timestamp("2024-01-03"), "SPY_close"] == 200
    assert dataset.loc[pd.Timestamp("2024-01-03"), "SPY_close"] != 300


def test_date_alignment_sorted_unique_timezone_naive() -> None:
    ff5_dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    market_dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    ff5 = pd.DataFrame(
        {
            "Mkt-RF": [0.03, 0.01, 0.02],
            "SMB": [0.03, 0.01, 0.02],
            "HML": [0.03, 0.01, 0.02],
            "RMW": [0.03, 0.01, 0.02],
            "CMA": [0.03, 0.01, 0.02],
            "RF": [0.0, 0.0, 0.0],
        },
        index=ff5_dates,
    )
    market = pd.DataFrame(
        {
            "SPY_open": [10, 20, 30],
            "SPY_high": [11, 21, 31],
            "SPY_low": [9, 19, 29],
            "SPY_close": [100, 200, 300],
            "SPY_volume": [1000, 2000, 3000],
        },
        index=market_dates,
    )

    dataset = build_modeling_dataset(ff5, market, _config())

    assert dataset.index.is_monotonic_increasing
    assert dataset.index.is_unique
    assert dataset.index.tz is None
