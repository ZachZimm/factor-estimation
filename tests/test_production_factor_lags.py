from __future__ import annotations

import pandas as pd

from ff5_predictor.nowcast_features import build_nowcast_features


def test_recursive_factor_lags_can_use_prior_predictions_without_hidden_actuals() -> None:
    dates = pd.date_range("2024-01-02", periods=4)
    ff5 = pd.DataFrame(
        {
            "Mkt-RF": [0.01, 0.02],
            "SMB": [0.01, 0.02],
            "HML": [0.01, 0.02],
            "RMW": [0.01, 0.02],
            "CMA": [0.01, 0.02],
            "RF": [0.0, 0.0],
        },
        index=dates[:2],
    )
    market = pd.DataFrame(
        {
            "SPY_open": [100, 101, 102, 103],
            "SPY_high": [100, 101, 102, 103],
            "SPY_low": [100, 101, 102, 103],
            "SPY_close": [100, 101, 102, 103],
            "SPY_volume": [1, 1, 1, 1],
        },
        index=dates,
    )
    config = {
        "prediction": {"target_columns": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]},
        "availability": {"market_data_lag_rows": 0, "official_factor_lag_rows": 1},
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
        "target_features": {"include_lagged_targets": True, "lags": [1], "rolling_windows": [], "include_rf_lags": True},
        "fundamentals": {"enabled": False},
    }
    recursive = pd.DataFrame({"date": ["2024-01-04"], "pred_Mkt-RF": [0.99], "pred_SMB": [0.0], "pred_HML": [0.0], "pred_RMW": [0.0], "pred_CMA": [0.0]})

    result = build_nowcast_features(
        ff5,
        market,
        config,
        official_cutoff_date=pd.Timestamp("2024-01-03"),
        recursive_predictions=recursive,
    )

    assert result.features.loc[pd.Timestamp("2024-01-04"), "Mkt-RF_lag_1"] == 0.02
    assert result.features.loc[pd.Timestamp("2024-01-05"), "Mkt-RF_lag_1"] == 0.99
    assert result.features.loc[pd.Timestamp("2024-01-05"), "RF_lag_1"] == 0.0
