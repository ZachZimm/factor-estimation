from __future__ import annotations

import pandas as pd

from ff5_predictor.target_features import build_lagged_target_features


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _ff5() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    data = {target: [float(i) for i in range(8)] for target in TARGETS}
    data["RF"] = [float(i) / 10 for i in range(8)]
    return pd.DataFrame(data, index=dates)


def test_lagged_target_features_use_prior_rows() -> None:
    cfg = {
        "prediction": {"horizon": 1, "target_columns": TARGETS},
        "target_features": {
            "include_lagged_targets": True,
            "lags": [1, 2],
            "rolling_windows": [3],
            "include_rolling_mean": True,
            "include_rolling_volatility": True,
            "include_rolling_min_max": True,
            "include_rf_lags": True,
        },
    }
    features = build_lagged_target_features(_ff5(), cfg)

    assert features.loc[pd.Timestamp("2024-01-04"), "Mkt-RF_lag_1"] == 2.0
    assert features.loc[pd.Timestamp("2024-01-04"), "Mkt-RF_lag_2"] == 1.0
    assert features.loc[pd.Timestamp("2024-01-04"), "Mkt-RF_mean_3"] == 1.0
    assert features.loc[pd.Timestamp("2024-01-04"), "RF_lag_1"] == 0.2


def test_rf_lags_are_optional() -> None:
    cfg = {
        "prediction": {"horizon": 1, "target_columns": TARGETS},
        "target_features": {
            "include_lagged_targets": True,
            "lags": [1],
            "rolling_windows": [],
            "include_rf_lags": False,
        },
    }
    features = build_lagged_target_features(_ff5(), cfg)

    assert "RF_lag_1" not in features.columns
