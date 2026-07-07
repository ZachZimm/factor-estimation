from __future__ import annotations

import numpy as np
import pandas as pd

from ff5_predictor.config import load_config
from ff5_predictor.elasticnet_validation import (
    make_time_series_validation_folds,
    run_elasticnet_validation_from_frames,
)


TARGETS = ["Mkt-RF", "SMB"]


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
            "lookback_windows": [1, 2, 5],
            "include_returns": True,
            "include_log_returns": True,
            "include_rolling_volatility": True,
            "include_rolling_volume_features": False,
            "include_ohlc_features": True,
            "include_drawdown": False,
            "drop_raw_ohlcv": True,
        },
        "proxy_features": {"enabled": False},
        "target_features": {"include_lagged_targets": False, "lags": [], "rolling_windows": []},
        "feature_extraction": {"enabled": False, "method": "none", "apply_to_models": []},
        "fundamentals": {"enabled": False},
        "nowcast": {
            "output_dir": str(tmp_path),
            "run_name": "elasticnet_validation_test",
            "models": ["elasticnet", "per_factor_elasticnet"],
            "primary_model": "per_factor_elasticnet",
            "train_window_rows": 12,
            "min_train_rows": 8,
        },
        "elasticnet_validation": {
            "protocols": ["sliding", "expanding"],
            "validation_start_date": "2020-01-24",
            "train_window_rows": 12,
            "min_train_rows": 8,
            "validation_window_rows": 4,
            "fold_step_rows": 4,
            "holdout_rows": 4,
            "vintage_step_rows": 4,
            "max_vintages": 3,
            "nonzero_threshold": 1.0e-12,
            "top_n_features": 3,
            "fixed_hyperparameters": {
                "alpha": 0.001,
                "l1_ratio": 0.05,
                "max_iter": 5000,
                "tol": 0.001,
                "scale_features": True,
                "tune_alpha": False,
                "tune_l1_ratio": False,
            },
        },
        "models": {
            "elasticnet": {
                "alpha": 0.001,
                "alpha_grid": [0.001],
                "l1_ratio": 0.05,
                "max_iter": 5000,
                "tol": 0.001,
                "tune_alpha": False,
                "tune_l1_ratio": False,
                "scale_features": True,
            },
            "per_factor_elasticnet": {
                "alpha": 0.001,
                "alpha_grid": [0.001],
                "l1_ratio": 0.05,
                "max_iter": 5000,
                "tol": 0.001,
                "tune_alpha": False,
                "tune_l1_ratio": False,
                "scale_features": True,
            },
        },
        "output": {"force_overwrite": False},
    }


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", periods=44)
    x = np.linspace(0.0, 3.0, len(dates))
    close = 100.0 + np.cumsum(0.2 + np.sin(x) * 0.1)
    ret = pd.Series(close, index=dates).pct_change().fillna(0.0).to_numpy()
    ff5 = pd.DataFrame(
        {
            "Mkt-RF": 0.4 * ret + 0.0002 * np.cos(x),
            "SMB": -0.2 * ret + 0.0001 * np.sin(2 * x),
            "RF": 0.0,
        },
        index=dates,
    )
    market = pd.DataFrame(
        {
            "SPY_open": close * 0.999,
            "SPY_high": close * 1.002,
            "SPY_low": close * 0.998,
            "SPY_close": close,
            "SPY_volume": np.linspace(1000, 2000, len(dates)),
        },
        index=dates,
    )
    return ff5, market


def test_validation_fold_construction_is_time_ordered() -> None:
    dates = pd.bdate_range("2020-01-01", periods=44)
    cfg = _config("/tmp")
    folds = make_time_series_validation_folds(dates, cfg)

    assert folds
    sliding = [fold for fold in folds if fold.protocol == "sliding"]
    expanding = [fold for fold in folds if fold.protocol == "expanding"]
    assert sliding and expanding
    assert all(len(fold.train_positions) <= 12 for fold in sliding)
    assert [len(fold.train_positions) for fold in expanding] == sorted(
        len(fold.train_positions) for fold in expanding
    )
    for fold in folds:
        assert fold.train_end_date < fold.validation_start_date
        assert max(fold.train_positions) < min(fold.validation_positions)


def test_elasticnet_validation_outputs_predictions_coefficients_and_metadata(tmp_path) -> None:
    ff5, market = _frames()
    result = run_elasticnet_validation_from_frames(ff5, market, _config(tmp_path))

    assert not result.fold_predictions.empty
    assert not result.vintage_predictions.empty
    assert not result.coefficient_table.empty
    assert not result.coefficient_stability.empty
    assert not result.vintage_stability.empty
    assert {"elasticnet", "per_factor_elasticnet"}.issubset(set(result.fold_predictions["model_type"]))
    assert {"pred_Mkt-RF", "actual_Mkt-RF", "error_Mkt-RF"}.issubset(result.fold_predictions.columns)

    assert set(result.coefficient_table["target"]) == set(TARGETS)
    assert set(result.coefficient_table["model_type"]) == {"elasticnet", "per_factor_elasticnet"}
    assert np.isfinite(result.coefficient_table["coefficient"].astype(float)).all()
    assert (result.coefficient_table["alpha"].astype(float) == 0.001).all()
    assert (result.coefficient_table["l1_ratio"].astype(float) == 0.05).all()
    assert (result.coefficient_table["tune_alpha"] == False).all()  # noqa: E712

    expected_paths = [
        "predictions/fold_predictions.csv",
        "predictions/vintage_holdout_predictions.csv",
        "tables/fold_metrics.csv",
        "tables/coefficient_table.csv",
        "tables/coefficient_stability.csv",
        "tables/vintage_stability.csv",
        "tables/model_summary.csv",
        "figures/fold_rmse_over_time.svg",
        "figures/coefficient_stability_over_folds.svg",
        "figures/top_feature_coefficient_paths.svg",
        "figures/vintage_holdout_rmse_vs_staleness.svg",
        "validation_report.html",
        "metadata/validation_metadata.json",
    ]
    for relative_path in expected_paths:
        assert (result.run_dir / relative_path).exists()


def test_vintage_holdout_uses_same_dates_and_ordered_staleness(tmp_path) -> None:
    ff5, market = _frames()
    result = run_elasticnet_validation_from_frames(ff5, market, _config(tmp_path))
    vintages = result.vintage_predictions

    holdout_dates = None
    for _, group in vintages.groupby(["protocol", "model_type", "vintage_id"]):
        dates = tuple(group["target_date"].tolist())
        holdout_dates = dates if holdout_dates is None else holdout_dates
        assert dates == holdout_dates
        assert pd.to_datetime(group["train_end_date"]).max() < pd.to_datetime(group["target_date"]).min()

    staleness = sorted(vintages["staleness_rows"].astype(int).unique())
    assert staleness == sorted(staleness)
    assert staleness[0] == 0
    assert len(staleness) > 1


def test_validation_config_and_script_exist() -> None:
    config = load_config("config/research/elasticnet_time_series_validation.yaml")
    assert config["nowcast"]["models"] == ["elasticnet", "per_factor_elasticnet"]
    assert config["models"]["elasticnet"]["tune_alpha"] is False
    assert config["models"]["per_factor_elasticnet"]["tune_alpha"] is False
    assert config["elasticnet_validation"]["fixed_hyperparameters"]["alpha"] == 0.001
