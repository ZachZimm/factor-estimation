from __future__ import annotations

import pandas as pd

from ff5_predictor.release_gap_backtest import run_release_gap_backtest_from_frames


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _config(tmp_path) -> dict:
    return {
        "data": {"tickers": ["SPY"]},
        "prediction": {"target_columns": TARGETS},
        "availability": {
            "market_data_lag_rows": 0,
            "official_factor_lag_rows": 1,
            "recursive_factor_lags": True,
            "release_gap_backtest_days": [1, 2],
        },
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
        "target_features": {"include_lagged_targets": True, "lags": [1], "rolling_windows": []},
        "fundamentals": {"enabled": False},
        "nowcast": {
            "output_dir": str(tmp_path),
            "run_name": "gap_test",
            "models": ["rolling_mean", "ewma", "ridge"],
            "train_window_rows": 10,
            "min_train_rows": 3,
            "backtest_step_rows": 2,
        },
        "models": {
            "ridge": {"alpha": 1.0, "tune_alpha": False, "scale_features": True},
            "ewma": {"default_span": 2},
        },
        "output": {"force_overwrite": False},
    }


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-02", periods=10)
    ff5 = pd.DataFrame({col: [0.001 * (i + j + 1) for i in range(10)] for j, col in enumerate(TARGETS)}, index=dates)
    ff5["RF"] = 0.0
    close = [100 + i for i in range(10)]
    market = pd.DataFrame(
        {
            "SPY_open": close,
            "SPY_high": close,
            "SPY_low": close,
            "SPY_close": close,
            "SPY_volume": [1] * 10,
        },
        index=dates,
    )
    return ff5, market


def test_release_gap_backtest_outputs_gap_metadata_and_metrics(tmp_path) -> None:
    ff5, market = _frames()
    result = run_release_gap_backtest_from_frames(ff5, market, _config(tmp_path))

    assert not result.predictions.empty
    assert {"cutoff_date", "target_date", "gap_day", "release_gap_size", "actual_Mkt-RF"}.issubset(result.predictions.columns)
    assert (pd.to_datetime(result.predictions["target_date"]) > pd.to_datetime(result.predictions["cutoff_date"])).all()
    assert "metrics_by_gap_day" in result.gap_metrics
    assert (result.run_dir / "metrics" / "model_ranking.csv").exists()


def test_release_gap_backtest_parallel_cutoffs(tmp_path) -> None:
    config = _config(tmp_path)
    config["nowcast"]["run_name"] = "gap_parallel_test"
    config["backtest"] = {"n_jobs": 2, "backend": "loky", "verbose": 0}
    ff5, market = _frames()
    result = run_release_gap_backtest_from_frames(ff5, market, config)

    assert not result.predictions.empty
    metadata = (result.run_dir / "metadata" / "backtest_metadata.json").read_text()
    assert '"backtest_n_jobs": 2' in metadata
