from __future__ import annotations

import pandas as pd

from ff5_predictor.model_implied_series import run_model_implied_series_from_frames


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _config(tmp_path) -> dict:
    return {
        "data": {"tickers": ["SPY"]},
        "prediction": {"target_columns": TARGETS},
        "availability": {
            "market_data_lag_rows": 0,
            "official_factor_lag_rows": 1,
            "recursive_factor_lags": False,
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
        "target_features": {"include_lagged_targets": False, "lags": [], "rolling_windows": []},
        "fundamentals": {"enabled": False},
        "nowcast": {
            "output_dir": str(tmp_path),
            "run_name": "model_implied_test",
            "models": ["rolling_mean", "ewma", "ridge"],
            "primary_model": "ridge",
            "train_window_rows": 10,
            "min_train_rows": 3,
        },
        "model_implied_series": {"refit_step_rows": 2, "n_jobs": 1},
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


def test_model_implied_series_saves_walk_forward_predictions(tmp_path) -> None:
    ff5, market = _frames()
    result = run_model_implied_series_from_frames(ff5, market, _config(tmp_path))

    assert not result.predictions.empty
    assert {"pred_Mkt-RF", "actual_Mkt-RF", "error_Mkt-RF", "cutoff_date", "target_date"}.issubset(
        result.predictions.columns
    )
    assert (pd.to_datetime(result.predictions["target_date"]) > pd.to_datetime(result.predictions["cutoff_date"])).all()
    assert (result.run_dir / "predictions" / "model_implied_ff5_series.csv").exists()
    assert (result.run_dir / "predictions" / "model_implied_factor_series.csv").exists()
    assert (result.run_dir / "predictions" / "official_minus_model_implied_series.csv").exists()
    assert (result.run_dir / "metrics" / "error_summary.csv").exists()
    metadata = (result.run_dir / "metadata" / "model_implied_series_metadata.json").read_text()
    assert "walk_forward_model_implied_ff5_series" in metadata


def test_model_implied_series_respects_date_filter(tmp_path) -> None:
    config = _config(tmp_path)
    config["nowcast"]["run_name"] = "model_implied_date_filter_test"
    config["date_filter"] = {"start_date": "2024-01-08", "end_date": "2024-01-09"}
    ff5, market = _frames()
    result = run_model_implied_series_from_frames(ff5, market, config)

    assert not result.predictions.empty
    assert set(pd.to_datetime(result.predictions["target_date"]).dt.date) == {
        pd.Timestamp("2024-01-08").date(),
        pd.Timestamp("2024-01-09").date(),
    }
