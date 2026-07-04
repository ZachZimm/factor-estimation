from __future__ import annotations

import pandas as pd

from ff5_predictor.latest_nowcast import run_latest_nowcast_from_frames


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _config(tmp_path) -> dict:
    return {
        "data": {"tickers": ["SPY"]},
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
        "target_features": {"include_lagged_targets": True, "lags": [1], "rolling_windows": []},
        "fundamentals": {"enabled": False},
        "nowcast": {
            "output_dir": str(tmp_path),
            "run_name": "test_nowcast",
            "models": ["ridge"],
            "primary_model": "ridge",
            "train_window_rows": 10,
            "save_feature_snapshot": True,
            "save_model_artifact": True,
        },
        "models": {"ridge": {"alpha": 1.0, "tune_alpha": False, "scale_features": True}},
        "output": {"force_overwrite": False},
    }


def _ff5(n: int = 6) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n)
    data = {col: [0.001 * (i + j + 1) for i in range(n)] for j, col in enumerate(TARGETS)}
    data["RF"] = [0.0] * n
    return pd.DataFrame(data, index=dates)


def _market(n: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n)
    close = [100 + i for i in range(n)]
    return pd.DataFrame(
        {
            "SPY_open": close,
            "SPY_high": close,
            "SPY_low": close,
            "SPY_close": close,
            "SPY_volume": [1] * n,
        },
        index=dates,
    )


def test_latest_nowcast_writes_predictions_and_model_artifact(tmp_path) -> None:
    result = run_latest_nowcast_from_frames(_ff5(), _market(), _config(tmp_path))

    assert not result.predictions.empty
    assert {"pred_Mkt-RF", "factor_data_asof", "market_data_asof"}.issubset(result.predictions.columns)
    assert (result.run_dir / "models" / "ridge.joblib").exists()
    assert (result.run_dir / "predictions" / "latest_nowcast.csv").exists()
    assert (result.run_dir / "predictions" / "official_plus_nowcast_series.csv").exists()
    assert not (tmp_path / "test_nowcast").joinpath("latest").exists()


def test_latest_nowcast_writes_elasticnet_attributions(tmp_path) -> None:
    cfg = _config(tmp_path)
    cfg["nowcast"]["models"] = ["elasticnet"]
    cfg["nowcast"]["primary_model"] = "elasticnet"
    cfg["nowcast"]["save_feature_attributions"] = True
    cfg["models"]["elasticnet"] = {
        "alpha": 0.001,
        "l1_ratio": 0.05,
        "max_iter": 10000,
        "tol": 0.0001,
        "tune_alpha": False,
        "scale_features": True,
    }

    result = run_latest_nowcast_from_frames(_ff5(7), _market(8), cfg)

    assert not result.predictions.empty
    assert (result.run_dir / "models" / "elasticnet.joblib").exists()
    assert (result.run_dir / "attribution" / "elasticnet_coefficients.csv").exists()
    assert (result.run_dir / "attribution" / "elasticnet_top_contributions.csv").exists()
    assert result.metadata["attribution"]["enabled"] is True
    assert result.metadata["attribution"]["model_type"] == "elasticnet"


def test_latest_nowcast_writes_empty_when_no_unreleased_dates(tmp_path) -> None:
    result = run_latest_nowcast_from_frames(_ff5(6), _market(6), _config(tmp_path))

    assert result.predictions.empty
    assert "pred_Mkt-RF" in result.predictions.columns
    assert (result.run_dir / "predictions" / "latest_nowcast.csv").exists()


def test_latest_nowcast_filters_prediction_dates(tmp_path) -> None:
    cfg = _config(tmp_path)
    cfg["date_filter"] = {"start_date": "2024-01-09", "end_date": "2024-01-09"}
    result = run_latest_nowcast_from_frames(_ff5(6), _market(8), cfg)

    assert not result.predictions.empty
    assert set(pd.to_datetime(result.predictions["date"])) == {pd.Timestamp("2024-01-09")}
    assert set(result.predictions["gap_day"]) == {2}
    assert result.metadata["n_all_unreleased_dates"] == 2
    assert result.metadata["n_unreleased_dates"] == 1


def test_latest_nowcast_tft_smoke(tmp_path) -> None:
    cfg = _config(tmp_path)
    cfg["nowcast"]["models"] = ["tft"]
    cfg["nowcast"]["save_model_artifact"] = False
    cfg["sequence"] = {"lookback_rows": 2, "batch_size": 4, "num_workers": 0}
    cfg["torch"] = {
        "device": "cpu",
        "max_epochs": 1,
        "patience": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "standardize_targets": True,
        "restore_best_checkpoint": True,
        "gradient_clip_norm": 1.0,
    }
    cfg["models"]["tft"] = {"hidden_size": 8, "n_heads": 1, "lstm_layers": 1, "dropout": 0.0}

    result = run_latest_nowcast_from_frames(_ff5(7), _market(8), cfg)

    assert not result.predictions.empty
    assert set(result.predictions["model_type"]) == {"tft"}
