from __future__ import annotations

import pandas as pd

from ff5_predictor.nowcast_engine import NowcastTargetSpec, run_nowcast_engine, select_backtest_columns, select_production_columns
from ff5_predictor.nowcast_features import build_nowcast_features


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _config() -> dict:
    return {
        "data": {"tickers": ["SPY"]},
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
        "nowcast": {"models": ["rolling_mean", "ewma", "ridge"], "train_window_rows": 10, "min_train_rows": 3},
        "models": {
            "ridge": {"alpha": 1.0, "tune_alpha": False, "scale_features": True},
            "ewma": {"default_span": 2},
        },
    }


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-02", periods=8)
    ff5 = pd.DataFrame({col: [0.001 * (i + j + 1) for i in range(8)] for j, col in enumerate(TARGETS)}, index=dates)
    ff5["RF"] = 0.0
    close = [100 + i for i in range(8)]
    market = pd.DataFrame(
        {
            "SPY_open": close,
            "SPY_high": close,
            "SPY_low": close,
            "SPY_close": close,
            "SPY_volume": [1] * 8,
        },
        index=dates,
    )
    return ff5, market


def _train_frame(ff5: pd.DataFrame, market: pd.DataFrame, cutoff: pd.Timestamp, config: dict):
    feature_result = build_nowcast_features(ff5.loc[:cutoff], market.loc[:cutoff], config)
    train_df = feature_result.features.join(ff5[TARGETS].loc[:cutoff], how="inner").dropna()
    return train_df, feature_result.feature_columns


def test_engine_production_and_backtest_schemas() -> None:
    ff5, market = _frames()
    config = _config()
    cutoff = pd.Timestamp("2024-01-05")
    train_df, feature_columns = _train_frame(ff5, market, cutoff, config)
    target_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-07")])

    production = run_nowcast_engine(
        ff5,
        market,
        train_df,
        feature_columns,
        TARGETS,
        NowcastTargetSpec(target_dates, cutoff, market.index.max(), None, True),
        config,
    )
    production_predictions = select_production_columns(production.predictions, TARGETS)

    assert "actual_Mkt-RF" not in production_predictions.columns
    assert {"pred_Mkt-RF", "market_data_asof", "factor_data_asof"}.issubset(production_predictions.columns)
    assert set(production_predictions["model_type"]) == {"rolling_mean", "ewma", "ridge"}
    assert not production.feature_snapshots.empty

    backtest = run_nowcast_engine(
        ff5,
        market,
        train_df,
        feature_columns,
        TARGETS,
        NowcastTargetSpec(
            target_dates,
            cutoff,
            target_dates.max(),
            ff5.loc[target_dates, TARGETS],
            False,
            {pd.Timestamp("2024-01-06"): [1, 2], pd.Timestamp("2024-01-07"): [2]},
        ),
        config,
    )
    backtest_predictions = select_backtest_columns(backtest.predictions, TARGETS)

    assert {"cutoff_date", "target_date", "release_gap_size", "actual_Mkt-RF"}.issubset(backtest_predictions.columns)
    assert not backtest_predictions.empty
    assert (backtest_predictions["recursive_factor_lags"] == False).all()  # noqa: E712
