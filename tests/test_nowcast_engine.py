from __future__ import annotations

import pandas as pd
import pytest

import ff5_predictor.nowcast_engine as nowcast_engine
from ff5_predictor.nowcast_engine import NowcastTargetSpec, run_nowcast_engine, select_backtest_columns, select_nowcast_columns
from ff5_predictor.nowcast_features import build_nowcast_features


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
TARGETS_WITH_MOM = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


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


def test_engine_latest_and_backtest_schemas() -> None:
    ff5, market = _frames()
    config = _config()
    cutoff = pd.Timestamp("2024-01-05")
    train_df, feature_columns = _train_frame(ff5, market, cutoff, config)
    target_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-07")])

    nowcast_result = run_nowcast_engine(
        ff5,
        market,
        train_df,
        feature_columns,
        TARGETS,
        NowcastTargetSpec(target_dates, cutoff, market.index.max(), None, True),
        config,
    )
    nowcast_predictions = select_nowcast_columns(nowcast_result.predictions, TARGETS)

    assert "actual_Mkt-RF" not in nowcast_predictions.columns
    assert {"pred_Mkt-RF", "market_data_asof", "factor_data_asof"}.issubset(nowcast_predictions.columns)
    assert set(nowcast_predictions["model_type"]) == {"rolling_mean", "ewma", "ridge"}
    assert not nowcast_result.feature_snapshots.empty

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


def test_engine_ridge_with_group_pca_and_pls() -> None:
    ff5, market = _frames()
    cutoff = pd.Timestamp("2024-01-06")
    target_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-07")])
    for method in ["group_pca", "pls"]:
        config = _config()
        config["nowcast"] = {"models": ["ridge"], "train_window_rows": 10, "min_train_rows": 3}
        config["feature_extraction"] = {
            "enabled": True,
            "method": method,
            "apply_to_models": ["ridge"],
            "group_pca": {
                "scale_before_pca": True,
                "groups": {"market_returns": {"patterns": ["*_ret_1d"], "n_components": 1}},
            },
            "pls": {"n_components": 2, "scale_features": True, "scale_targets": False},
        }
        train_df, feature_columns = _train_frame(ff5, market, cutoff, config)
        result = run_nowcast_engine(
            ff5,
            market,
            train_df,
            feature_columns,
            TARGETS,
            NowcastTargetSpec(target_dates, cutoff, target_dates.max(), ff5.loc[target_dates, TARGETS], False),
            config,
        )
        predictions = select_backtest_columns(result.predictions, TARGETS)
        assert not predictions.empty
        assert set(predictions["feature_extraction_method"]) == {method}


def test_engine_ridge_with_hybrid_raw_and_extracted_features() -> None:
    ff5, market = _frames()
    config = _config()
    config["nowcast"] = {"models": ["ridge"], "train_window_rows": 10, "min_train_rows": 3}
    config["feature_extraction"] = {
        "enabled": True,
        "method": "group_pca",
        "apply_to_models": ["ridge"],
        "keep_original_features": True,
        "group_pca": {
            "scale_before_pca": True,
            "groups": {"market_returns": {"patterns": ["*_ret_1d"], "n_components": 1}},
        },
    }
    cutoff = pd.Timestamp("2024-01-06")
    target_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-07")])
    train_df, feature_columns = _train_frame(ff5, market, cutoff, config)
    result = run_nowcast_engine(
        ff5,
        market,
        train_df,
        feature_columns,
        TARGETS,
        NowcastTargetSpec(target_dates, cutoff, target_dates.max(), ff5.loc[target_dates, TARGETS], False),
        config,
    )
    predictions = select_backtest_columns(result.predictions, TARGETS)
    assert not predictions.empty
    assert int(predictions["n_model_features"].iloc[0]) > int(predictions["n_raw_features"].iloc[0])


def test_engine_per_target_pls_ridge() -> None:
    ff5, market = _frames()
    config = _config()
    config["nowcast"] = {"models": ["per_target_pls_ridge"], "train_window_rows": 10, "min_train_rows": 3}
    config["feature_extraction"] = {
        "enabled": True,
        "method": "per_target_pls",
        "apply_to_models": ["per_target_pls_ridge"],
        "per_target_pls": {"n_components": 2, "scale_features": True, "scale_targets": False},
    }
    cutoff = pd.Timestamp("2024-01-06")
    target_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-07")])
    train_df, feature_columns = _train_frame(ff5, market, cutoff, config)
    result = run_nowcast_engine(
        ff5,
        market,
        train_df,
        feature_columns,
        TARGETS,
        NowcastTargetSpec(target_dates, cutoff, target_dates.max(), ff5.loc[target_dates, TARGETS], False),
        config,
    )
    predictions = select_backtest_columns(result.predictions, TARGETS)
    assert set(predictions["model_type"]) == {"per_target_pls_ridge"}
    assert "pred_Mkt-RF" in predictions.columns


def test_engine_elasticnet_smoke() -> None:
    ff5, market = _frames()
    config = _config()
    config["nowcast"] = {"models": ["elasticnet"], "train_window_rows": 10, "min_train_rows": 3}
    config["models"]["elasticnet"] = {
        "alpha": 0.01,
        "alpha_grid": [0.01],
        "l1_ratio": 0.05,
        "max_iter": 1000,
        "tol": 0.001,
        "tune_alpha": False,
        "scale_features": True,
    }
    cutoff = pd.Timestamp("2024-01-06")
    target_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-07")])
    train_df, feature_columns = _train_frame(ff5, market, cutoff, config)
    result = run_nowcast_engine(
        ff5,
        market,
        train_df,
        feature_columns,
        TARGETS,
        NowcastTargetSpec(target_dates, cutoff, target_dates.max(), ff5.loc[target_dates, TARGETS], False),
        config,
    )
    predictions = select_backtest_columns(result.predictions, TARGETS)
    assert set(predictions["model_type"]) == {"elasticnet"}
    assert "pred_Mkt-RF" in predictions.columns


def test_engine_gradient_boosting_smoke() -> None:
    ff5, market = _frames()
    config = _config()
    config["nowcast"] = {"models": ["gradient_boosting"], "train_window_rows": 10, "min_train_rows": 3}
    config["models"]["gradient_boosting"] = {
        "learning_rate": 0.1,
        "max_iter": 5,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 2,
        "early_stopping": False,
        "n_jobs": 1,
    }
    cutoff = pd.Timestamp("2024-01-06")
    target_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-07")])
    train_df, feature_columns = _train_frame(ff5, market, cutoff, config)
    result = run_nowcast_engine(
        ff5,
        market,
        train_df,
        feature_columns,
        TARGETS,
        NowcastTargetSpec(target_dates, cutoff, target_dates.max(), ff5.loc[target_dates, TARGETS], False),
        config,
    )
    predictions = select_backtest_columns(result.predictions, TARGETS)
    assert set(predictions["model_type"]) == {"gradient_boosting"}
    assert "pred_Mkt-RF" in predictions.columns


def test_engine_uses_precomputed_feature_frame_without_rebuilding(monkeypatch: pytest.MonkeyPatch) -> None:
    ff5, market = _frames()
    config = _config()
    config["nowcast"] = {"models": ["ridge"], "train_window_rows": 10, "min_train_rows": 3}
    cutoff = pd.Timestamp("2024-01-06")
    target_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-07")])
    full_features = build_nowcast_features(ff5.iloc[0:0], market, config).features
    train_features = full_features.loc[:cutoff].dropna(axis=1, how="all")
    train_df = train_features.join(ff5[TARGETS].loc[:cutoff]).dropna()

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("build_nowcast_features should not be called when feature_frame is supplied")

    monkeypatch.setattr(nowcast_engine, "build_nowcast_features", fail_rebuild)
    result = run_nowcast_engine(
        ff5,
        market,
        train_df,
        list(train_features.columns),
        TARGETS,
        NowcastTargetSpec(target_dates, cutoff, target_dates.max(), ff5.loc[target_dates, TARGETS], False),
        config,
        feature_frame=full_features,
    )
    predictions = select_backtest_columns(result.predictions, TARGETS)
    assert set(predictions["model_type"]) == {"ridge"}
    assert "pred_Mkt-RF" in predictions.columns


def test_engine_per_factor_elasticnet_smoke() -> None:
    ff5, market = _frames()
    config = _config()
    config["nowcast"] = {"models": ["per_factor_elasticnet"], "train_window_rows": 10, "min_train_rows": 3}
    config["models"]["per_factor_elasticnet"] = {
        "alpha": 0.01,
        "alpha_grid": [0.01],
        "l1_ratio": 0.05,
        "max_iter": 1000,
        "tol": 0.001,
        "tune_alpha": False,
        "scale_features": True,
    }
    cutoff = pd.Timestamp("2024-01-06")
    target_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-07")])
    train_df, feature_columns = _train_frame(ff5, market, cutoff, config)
    result = run_nowcast_engine(
        ff5,
        market,
        train_df,
        feature_columns,
        TARGETS,
        NowcastTargetSpec(target_dates, cutoff, target_dates.max(), ff5.loc[target_dates, TARGETS], False),
        config,
    )
    predictions = select_backtest_columns(result.predictions, TARGETS)
    assert set(predictions["model_type"]) == {"per_factor_elasticnet"}
    assert "pred_Mkt-RF" in predictions.columns


def test_engine_elasticnet_mom_override_smoke() -> None:
    ff5, market = _frames()
    ff5["Mom"] = [0.002 * (i + 1) for i in range(len(ff5))]
    config = _config()
    config["prediction"] = {"target_columns": TARGETS_WITH_MOM}
    config["nowcast"] = {"models": ["elasticnet_mom_override"], "train_window_rows": 10, "min_train_rows": 3}
    config["models"]["elasticnet"] = {
        "alpha": 0.01,
        "alpha_grid": [0.01],
        "l1_ratio": 0.05,
        "max_iter": 1000,
        "tol": 0.001,
        "tune_alpha": False,
        "scale_features": True,
    }
    config["models"]["elasticnet_mom_override"] = dict(config["models"]["elasticnet"])
    cutoff = pd.Timestamp("2024-01-06")
    target_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-07")])
    feature_result = build_nowcast_features(ff5.loc[:cutoff], market.loc[:cutoff], config)
    train_df = feature_result.features.join(ff5[TARGETS_WITH_MOM].loc[:cutoff]).dropna()
    result = run_nowcast_engine(
        ff5,
        market,
        train_df,
        feature_result.feature_columns,
        TARGETS_WITH_MOM,
        NowcastTargetSpec(target_dates, cutoff, target_dates.max(), ff5.loc[target_dates, TARGETS_WITH_MOM], False),
        config,
    )
    predictions = select_backtest_columns(result.predictions, TARGETS_WITH_MOM)
    assert set(predictions["model_type"]) == {"elasticnet_mom_override"}
    assert "pred_Mom" in predictions.columns


def test_engine_tft_with_group_pca_smoke() -> None:
    ff5, market = _frames()
    config = _config()
    config["nowcast"] = {"models": ["tft"], "train_window_rows": 10, "min_train_rows": 3}
    config["feature_extraction"] = {
        "enabled": True,
        "method": "group_pca",
        "apply_to_models": ["tft"],
        "group_pca": {
            "scale_before_pca": True,
            "groups": {"market_returns": {"patterns": ["*_ret_1d"], "n_components": 1}},
        },
    }
    config["sequence"] = {"lookback_rows": 2, "batch_size": 2}
    config["torch"] = {"max_epochs": 1, "patience": 1, "device": "cpu", "standardize_targets": True}
    config["models"]["tft"] = {"hidden_size": 4, "n_heads": 1, "lstm_layers": 1, "dropout": 0.0}
    cutoff = pd.Timestamp("2024-01-07")
    target_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-08")])
    train_df, feature_columns = _train_frame(ff5, market, cutoff, config)
    result = run_nowcast_engine(
        ff5,
        market,
        train_df,
        feature_columns,
        TARGETS,
        NowcastTargetSpec(target_dates, cutoff, target_dates.max(), ff5.loc[target_dates, TARGETS], False),
        config,
    )
    predictions = select_backtest_columns(result.predictions, TARGETS)
    assert set(predictions["model_type"]) == {"tft"}
    assert set(predictions["feature_extraction_method"]) == {"group_pca"}
    assert not result.training_history.empty
    assert {"train_loss", "validation_rmse", "train_directional_accuracy"}.issubset(result.training_history.columns)
