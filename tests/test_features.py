from __future__ import annotations

import numpy as np
import pandas as pd

from ff5_predictor.features import build_market_features


def test_feature_calculations_are_backward_looking() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    market = pd.DataFrame(
        {
            "SPY_open": [10, 11, 12, 13, 14],
            "SPY_high": [11, 12, 13, 14, 15],
            "SPY_low": [9, 10, 11, 12, 13],
            "SPY_close": [10, 12, 15, 15, 30],
            "SPY_volume": [100, 110, 120, 130, 140],
        },
        index=dates,
    )
    config = {
        "features": {
            "lookback_windows": [2],
            "include_returns": True,
            "include_log_returns": True,
            "include_rolling_volatility": True,
            "include_rolling_volume_features": False,
            "include_ohlc_features": True,
            "include_drawdown": True,
            "drop_raw_ohlcv": True,
        }
    }

    features = build_market_features(market, config)

    assert features.loc[dates[3], "SPY_ret_2d"] == 15 / 12 - 1
    expected_vol = pd.Series([12 / 10 - 1, 15 / 12 - 1]).std(ddof=0)
    assert np.isclose(features.loc[dates[2], "SPY_vol_2d"], expected_vol)


def test_feature_output_has_expected_columns() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    market = pd.DataFrame(
        {
            "SPY_open": [10, 11, 12, 13],
            "SPY_high": [11, 12, 13, 14],
            "SPY_low": [9, 10, 11, 12],
            "SPY_close": [10, 12, 15, 16],
            "SPY_volume": [100, 110, 120, 130],
        },
        index=dates,
    )
    config = {
        "features": {
            "lookback_windows": [1, 2],
            "include_returns": True,
            "include_log_returns": True,
            "include_rolling_volatility": True,
            "include_rolling_volume_features": True,
            "include_ohlc_features": True,
            "include_drawdown": True,
            "drop_raw_ohlcv": False,
        }
    }

    features = build_market_features(market, config)

    for column in [
        "SPY_ret_1d",
        "SPY_log_ret_1d",
        "SPY_oc_ret",
        "SPY_hl_range",
        "SPY_gap",
        "SPY_ret_2d",
        "SPY_vol_2d",
        "SPY_volume_mean_2d",
        "SPY_volume_z_2d",
        "SPY_hl_range_mean_2d",
        "SPY_drawdown_2d",
        "SPY_close",
    ]:
        assert column in features.columns
    assert "SPY_volume_z_1d" not in features.columns
    assert features["SPY_vol_1d"].notna().any()


def test_all_nan_volume_z_features_are_dropped() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    market = pd.DataFrame(
        {
            "^VIX_open": [10, 11, 12, 13, 14],
            "^VIX_high": [11, 12, 13, 14, 15],
            "^VIX_low": [9, 10, 11, 12, 13],
            "^VIX_close": [10, 12, 15, 16, 18],
            "^VIX_volume": [0, 0, 0, 0, 0],
        },
        index=dates,
    )
    config = {
        "features": {
            "lookback_windows": [2],
            "include_returns": True,
            "include_log_returns": True,
            "include_rolling_volatility": True,
            "include_rolling_volume_features": True,
            "include_ohlc_features": True,
            "include_drawdown": True,
            "drop_raw_ohlcv": False,
        }
    }

    features = build_market_features(market, config)

    assert "^VIX_volume_z_2d" not in features.columns
    assert "^VIX_volume_mean_2d" in features.columns
    assert not features.empty
