from __future__ import annotations

import pandas as pd
import numpy as np

from ff5_predictor.nowcast_dataset import build_nowcast_dataset


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _config(tmp_path=None) -> dict:
    return {
        "data": {"tickers": ["SPY"], "cache_dir": "data/cache"},
        "prediction": {"target_columns": TARGETS},
        "availability": {"market_data_lag_rows": 0, "official_factor_lag_rows": 1, "recursive_factor_lags": True},
        "features": {
            "lookback_windows": [1],
            "include_returns": True,
            "include_log_returns": False,
            "include_rolling_volatility": False,
            "include_rolling_volume_features": False,
            "include_ohlc_features": False,
            "include_drawdown": False,
            "drop_raw_ohlcv": True,
        },
        "proxy_features": {"enabled": False},
        "target_features": {
            "include_lagged_targets": True,
            "lags": [1],
            "rolling_windows": [],
            "include_rf_lags": True,
        },
        "fundamentals": {"enabled": False},
    }


def _ff5() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=3)
    return pd.DataFrame(
        {
            "Mkt-RF": [0.01, 0.02, 0.03],
            "SMB": [0.001, 0.002, 0.003],
            "HML": [0.004, 0.005, 0.006],
            "RMW": [0.007, 0.008, 0.009],
            "CMA": [0.010, 0.011, 0.012],
            "RF": [0.0001, 0.0001, 0.0001],
        },
        index=dates,
    )


def _market() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=4)
    close = [100, 110, 121, 133.1]
    return pd.DataFrame(
        {
            "SPY_open": close,
            "SPY_high": close,
            "SPY_low": close,
            "SPY_close": close,
            "SPY_volume": [1, 1, 1, 1],
        },
        index=dates,
    )


def test_nowcast_dataset_uses_same_day_market_but_prior_factor_values() -> None:
    dataset = build_nowcast_dataset(_ff5(), _market(), _config())

    assert np.isclose(dataset.train_df.loc[pd.Timestamp("2024-01-03"), "SPY_ret_1d"], 0.10)
    assert dataset.train_df.loc[pd.Timestamp("2024-01-03"), "Mkt-RF_lag_1"] == 0.01
    assert dataset.train_df.loc[pd.Timestamp("2024-01-03"), "Mkt-RF"] == 0.02
    assert list(dataset.unreleased_dates) == [pd.Timestamp("2024-01-05")]
    assert "SPY_close" not in dataset.feature_columns


def test_nowcast_dataset_filters_unreleased_dates() -> None:
    cfg = _config()
    cfg["date_filter"] = {"start_date": "2024-01-06", "end_date": "2024-01-06"}
    ff5 = _ff5()
    market = _market()
    market.loc[pd.Timestamp("2024-01-06")] = market.iloc[-1]

    dataset = build_nowcast_dataset(ff5, market.sort_index(), cfg)

    assert list(dataset.unreleased_dates) == [pd.Timestamp("2024-01-06")]
    assert dataset.metadata["n_all_unreleased_dates"] == 2
    assert dataset.metadata["n_unreleased_dates"] == 1


def test_sparse_market_rows_do_not_poison_later_nowcast_features() -> None:
    ff5 = _ff5()
    market = _market()
    sparse_holiday = pd.Timestamp("2024-01-05")
    valid_after_holiday = pd.Timestamp("2024-01-06")
    market.loc[sparse_holiday, ["SPY_open", "SPY_high", "SPY_low", "SPY_close"]] = pd.NA
    market.loc[valid_after_holiday] = {
        "SPY_open": 133.1,
        "SPY_high": 133.1,
        "SPY_low": 133.1,
        "SPY_close": 133.1,
        "SPY_volume": 1,
    }

    dataset = build_nowcast_dataset(ff5, market.sort_index(), _config())

    assert sparse_holiday in dataset.unreleased_dates
    assert valid_after_holiday in dataset.inference_df.index
    assert np.isclose(dataset.inference_df.loc[valid_after_holiday, "SPY_ret_1d"], 0.10)
