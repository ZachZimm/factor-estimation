from __future__ import annotations

import numpy as np
import pandas as pd

from ff5_predictor.feature_extraction import fit_feature_extractor
from ff5_predictor.nowcast_engine import NowcastTargetSpec, run_nowcast_engine
from ff5_predictor.nowcast_features import build_nowcast_features


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def test_group_pca_fit_excludes_target_row_extreme_value() -> None:
    train = pd.DataFrame(
        {
            "SPY_ret_1d": [1.0, 2.0, 3.0],
            "QQQ_ret_1d": [1.0, 2.0, 3.0],
            **{target: [0.0, 0.1, 0.2] for target in TARGETS},
        },
        index=pd.date_range("2024-01-02", periods=3),
    )
    hidden = pd.DataFrame({"SPY_ret_1d": [10_000.0], "QQQ_ret_1d": [10_000.0]}, index=[pd.Timestamp("2024-01-05")])
    config = {
        "feature_extraction": {
            "enabled": True,
            "method": "group_pca",
            "apply_to_models": ["ridge"],
            "group_pca": {
                "scale_before_pca": True,
                "groups": {"market_returns": {"patterns": ["*_ret_1d"], "n_components": 1}},
            },
        }
    }
    extractor = fit_feature_extractor(train, ["SPY_ret_1d", "QQQ_ret_1d"], TARGETS, config, model_type="ridge")
    assert extractor is not None
    scaler_mean = extractor.model["pipelines"]["market_returns"].named_steps["scaler"].mean_
    assert np.allclose(scaler_mean, [2.0, 2.0])
    assert not np.isclose(scaler_mean[0], hidden.iloc[0, 0])


def test_engine_refits_extractor_per_cutoff_without_hidden_actuals() -> None:
    dates = pd.date_range("2024-01-02", periods=9)
    ff5 = pd.DataFrame({target: np.linspace(0.0, 0.008, 9) for target in TARGETS}, index=dates)
    ff5["RF"] = 0.0
    market = pd.DataFrame(
        {
            "SPY_open": np.arange(100, 109),
            "SPY_high": np.arange(100, 109),
            "SPY_low": np.arange(100, 109),
            "SPY_close": np.arange(100, 109),
            "SPY_volume": [1] * 9,
        },
        index=dates,
    )
    config = {
        "prediction": {"target_columns": TARGETS},
        "availability": {"market_data_lag_rows": 0, "official_factor_lag_rows": 1, "recursive_factor_lags": False},
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
        "target_features": {"include_lagged_targets": False, "lags": [], "rolling_windows": []},
        "fundamentals": {"enabled": False},
        "nowcast": {"models": ["ridge"], "train_window_rows": 10},
        "models": {"ridge": {"alpha": 1.0, "tune_alpha": False, "scale_features": True}},
        "feature_extraction": {
            "enabled": True,
            "method": "group_pca",
            "apply_to_models": ["ridge"],
            "group_pca": {
                "scale_before_pca": True,
                "groups": {"market_returns": {"patterns": ["*_ret_1d"], "n_components": 1}},
            },
        },
    }
    for cutoff in [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-06")]:
        feature_result = build_nowcast_features(ff5.loc[:cutoff], market.loc[:cutoff], config)
        train_df = feature_result.features.join(ff5[TARGETS].loc[:cutoff]).dropna()
        result = run_nowcast_engine(
            ff5.loc[:cutoff],
            market.loc[: pd.Timestamp("2024-01-07")],
            train_df,
            feature_result.feature_columns,
            TARGETS,
            NowcastTargetSpec(pd.DatetimeIndex([pd.Timestamp("2024-01-07")]), cutoff, pd.Timestamp("2024-01-07"), None, False),
            config,
        )
        fitted = result.fitted_models["ridge"]
        assert fitted.metadata["feature_extraction"]["n_fit_rows"] == len(train_df)
