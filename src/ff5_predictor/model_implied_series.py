from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from ff5_predictor.availability import filter_dates_by_config
from ff5_predictor.data_famafrench import load_ff5
from ff5_predictor.data_yfinance import load_market_data
from ff5_predictor.evaluation import evaluate_on_shared_dates, evaluate_prediction_groups, rank_models
from ff5_predictor.io import ensure_dir, normalize_datetime_index
from ff5_predictor.nowcast_engine import NowcastTargetSpec, run_nowcast_engine, select_backtest_columns
from ff5_predictor.nowcast_features import build_nowcast_features
from ff5_predictor.nowcast_io import create_nowcast_run_dir, write_json, write_nowcast_predictions, write_yaml
from ff5_predictor.training_diagnostics import write_training_diagnostics


@dataclass(frozen=True)
class ModelImpliedSeriesResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    run_dir: Path
    training_history: pd.DataFrame


@dataclass(frozen=True)
class _BatchPredictionResult:
    predictions: pd.DataFrame
    training_history: pd.DataFrame


def run_model_implied_series(config: dict[str, Any]) -> ModelImpliedSeriesResult:
    ff5_df = load_ff5(config)
    market_df = load_market_data(config)
    return run_model_implied_series_from_frames(ff5_df, market_df, config)


def run_model_implied_series_from_frames(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: dict[str, Any],
) -> ModelImpliedSeriesResult:
    ff5 = normalize_datetime_index(ff5_df)
    market = normalize_datetime_index(market_df)
    target_columns = list(config["prediction"]["target_columns"])
    aligned_dates = pd.DatetimeIndex(ff5.index.intersection(market.index)).sort_values()
    if aligned_dates.empty:
        predictions = pd.DataFrame()
        training_history = pd.DataFrame()
    else:
        batches = _make_prediction_batches(aligned_dates, config)
        static_feature_frame = _build_static_feature_frame(ff5, market, config)
        series_cfg = config.get("model_implied_series", {})
        n_jobs = int(series_cfg.get("n_jobs", config.get("backtest", {}).get("n_jobs", 1)))
        if n_jobs == 1 or len(batches) <= 1:
            results = [_predict_batch(ff5, market, batch, config, static_feature_frame) for batch in batches]
        else:
            chunks = _chunk_batches(batches, n_jobs)
            results = Parallel(
                n_jobs=n_jobs,
                backend=str(series_cfg.get("backend", config.get("backtest", {}).get("backend", "loky"))),
                verbose=int(series_cfg.get("verbose", config.get("backtest", {}).get("verbose", 0))),
            )(
                delayed(_predict_batch_chunk)(ff5, market, chunk, config, static_feature_frame)
                for chunk in chunks
                if chunk
            )
        predictions = pd.concat([result.predictions for result in results], ignore_index=True) if results else pd.DataFrame()
        histories = [result.training_history for result in results if not result.training_history.empty]
        training_history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()

    predictions = select_backtest_columns(predictions, target_columns)
    predictions = _add_error_columns(predictions, target_columns)

    run_dir = create_nowcast_run_dir(config)
    write_yaml(run_dir / "config_resolved.yaml", config)
    write_nowcast_predictions(run_dir, "model_implied_ff5_series.csv", predictions)
    write_nowcast_predictions(run_dir, "model_implied_factor_series.csv", predictions)
    residual_series = _official_minus_model_implied_series(predictions, target_columns)
    write_nowcast_predictions(run_dir, "official_minus_model_implied_series.csv", residual_series)

    metrics = evaluate_prediction_groups(predictions, target_columns) if not predictions.empty else {}
    shared = evaluate_on_shared_dates(predictions, target_columns, baseline_model="ewma") if not predictions.empty else {}
    ranking = rank_models(shared)
    error_summary = _error_summary(predictions, target_columns)
    ensure_dir(run_dir / "metrics")
    ranking.to_csv(run_dir / "metrics" / "model_ranking.csv", index=False)
    error_summary.to_csv(run_dir / "metrics" / "error_summary.csv", index=False)
    write_json(run_dir / "metrics" / "metrics.json", metrics)
    write_json(run_dir / "metrics" / "shared_date_metrics.json", shared)
    training_diagnostics_metadata = write_training_diagnostics(run_dir, training_history)
    write_json(
        run_dir / "metadata" / "model_implied_series_metadata.json",
        {**_metadata(config, predictions, aligned_dates), "training_diagnostics": training_diagnostics_metadata},
    )
    return ModelImpliedSeriesResult(predictions=predictions, metrics=metrics, run_dir=run_dir, training_history=training_history)


def _make_prediction_batches(aligned_dates: pd.DatetimeIndex, config: dict[str, Any]) -> list[pd.DatetimeIndex]:
    min_rows = int(config.get("nowcast", {}).get("min_train_rows", 1000))
    series_cfg = config.get("model_implied_series", {})
    refit_step_rows = int(series_cfg.get("refit_step_rows", config.get("nowcast", {}).get("backtest_step_rows", 21)))
    if refit_step_rows < 1:
        raise ValueError("model_implied_series.refit_step_rows must be at least 1")
    candidate_dates = pd.DatetimeIndex(aligned_dates[min_rows:]).sort_values()
    target_dates = filter_dates_by_config(candidate_dates, config)
    if target_dates.empty:
        return []
    batches = []
    for start in range(0, len(target_dates), refit_step_rows):
        batch = pd.DatetimeIndex(target_dates[start : start + refit_step_rows]).sort_values()
        if not batch.empty:
            batches.append(batch)
    return batches


def _predict_batch(
    ff5: pd.DataFrame,
    market: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    static_feature_frame: pd.DataFrame | None = None,
) -> _BatchPredictionResult:
    if target_dates.empty:
        return _BatchPredictionResult(pd.DataFrame(), pd.DataFrame())
    target_columns = list(config["prediction"]["target_columns"])
    cutoff_date = _previous_aligned_date(pd.DatetimeIndex(ff5.index.intersection(market.index)).sort_values(), target_dates[0])
    train_df, feature_columns = _training_frame(ff5, market, cutoff_date, config, static_feature_frame)
    min_rows = int(config.get("nowcast", {}).get("min_train_rows", 1000))
    if len(train_df) < min_rows:
        return _BatchPredictionResult(pd.DataFrame(), pd.DataFrame())
    spec = NowcastTargetSpec(
        target_dates=target_dates,
        cutoff_date=pd.Timestamp(cutoff_date),
        latest_market_date=pd.Timestamp(target_dates.max()),
        actuals=ff5.loc[target_dates, target_columns],
        is_unreleased=False,
        release_gap_size_by_date=None,
    )
    result = run_nowcast_engine(
        ff5_df=ff5.loc[:cutoff_date],
        market_df=market.loc[: pd.Timestamp(target_dates.max())],
        train_df=train_df,
        feature_columns=feature_columns,
        target_columns=target_columns,
        spec=spec,
        config=config,
        feature_frame=static_feature_frame,
    )
    return _BatchPredictionResult(result.predictions, result.training_history)


def _predict_batch_chunk(
    ff5: pd.DataFrame,
    market: pd.DataFrame,
    batches: list[pd.DatetimeIndex],
    config: dict[str, Any],
    static_feature_frame: pd.DataFrame | None = None,
) -> _BatchPredictionResult:
    results = [_predict_batch(ff5, market, batch, config, static_feature_frame) for batch in batches]
    predictions = pd.concat([result.predictions for result in results], ignore_index=True) if results else pd.DataFrame()
    histories = [result.training_history for result in results if not result.training_history.empty]
    training_history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    return _BatchPredictionResult(predictions, training_history)


def _previous_aligned_date(aligned_dates: pd.DatetimeIndex, target_date: pd.Timestamp) -> pd.Timestamp:
    earlier = aligned_dates[aligned_dates < pd.Timestamp(target_date)]
    if earlier.empty:
        raise ValueError(f"No training date exists before target date {target_date.date()}")
    return pd.Timestamp(earlier[-1])


def _training_frame(
    ff5: pd.DataFrame,
    market: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    config: dict[str, Any],
    static_feature_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    target_columns = list(config["prediction"]["target_columns"])
    if static_feature_frame is not None:
        feature_frame = static_feature_frame.loc[:cutoff_date].dropna(axis=1, how="all")
        train_df = feature_frame.join(ff5[target_columns].loc[:cutoff_date], how="inner")
        feature_columns = list(feature_frame.columns)
    else:
        feature_result = build_nowcast_features(ff5.loc[:cutoff_date], market.loc[:cutoff_date], config)
        train_df = feature_result.features.join(ff5[target_columns].loc[:cutoff_date], how="inner")
        feature_columns = list(feature_result.feature_columns)
    train_df = train_df.dropna(subset=feature_columns + target_columns)
    return train_df, feature_columns


def _build_static_feature_frame(ff5: pd.DataFrame, market: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame | None:
    if not _can_use_static_feature_frame(config):
        return None
    feature_result = build_nowcast_features(ff5.iloc[0:0], market, config)
    return feature_result.features


def _can_use_static_feature_frame(config: dict[str, Any]) -> bool:
    return (
        not bool(config.get("target_features", {}).get("include_lagged_targets", False))
        and not bool(config.get("availability", {}).get("recursive_factor_lags", False))
        and not bool(config.get("fundamentals", {}).get("enabled", False))
    )


def _chunk_batches(batches: list[pd.DatetimeIndex], n_jobs: int) -> list[list[pd.DatetimeIndex]]:
    n_chunks = max(1, min(int(n_jobs), len(batches)))
    return [list(batches[i::n_chunks]) for i in range(n_chunks)]


def _add_error_columns(predictions: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    result = predictions.copy()
    for target in target_columns:
        error = result[f"pred_{target}"].astype(float) - result[f"actual_{target}"].astype(float)
        result[f"error_{target}"] = error
        result[f"abs_error_{target}"] = error.abs()
    return result


def _error_summary(predictions: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["model_type", "target", "mae", "rmse", "mean_error", "std_error"])
    rows = []
    for model_type, group in predictions.groupby("model_type"):
        for target in target_columns:
            error = group[f"error_{target}"].astype(float)
            rows.append(
                {
                    "model_type": model_type,
                    "target": target,
                    "mae": float(error.abs().mean()),
                    "rmse": float(np.sqrt(np.mean(error * error))),
                    "mean_error": float(error.mean()),
                    "std_error": float(error.std(ddof=0)),
                }
            )
    return pd.DataFrame(rows)


def _metadata(config: dict[str, Any], predictions: pd.DataFrame, aligned_dates: pd.DatetimeIndex) -> dict[str, Any]:
    series_cfg = config.get("model_implied_series", {})
    return {
        "methodology": "walk_forward_model_implied_ff5_series",
        "description": (
            "Each prediction uses same-day market data for the target date and a model fitted only on "
            "official FF5 labels from dates before the checkpoint cutoff."
        ),
        "models": list(config.get("nowcast", {}).get("models", [])),
        "primary_model": config.get("nowcast", {}).get("primary_model"),
        "refit_step_rows": int(series_cfg.get("refit_step_rows", config.get("nowcast", {}).get("backtest_step_rows", 21))),
        "exact_daily_refit": int(series_cfg.get("refit_step_rows", config.get("nowcast", {}).get("backtest_step_rows", 21))) == 1,
        "uses_same_day_market_data": int(config.get("availability", {}).get("market_data_lag_rows", 0)) == 0,
        "uses_ff5_input_features": bool(config.get("target_features", {}).get("include_lagged_targets", False)),
        "uses_recursive_factor_lags": bool(config.get("availability", {}).get("recursive_factor_lags", False)),
        "n_aligned_dates": int(len(aligned_dates)),
        "aligned_start_date": None if aligned_dates.empty else pd.Timestamp(aligned_dates.min()).date().isoformat(),
        "aligned_end_date": None if aligned_dates.empty else pd.Timestamp(aligned_dates.max()).date().isoformat(),
        "date_filter": dict(config.get("date_filter", {})),
        "n_prediction_rows": int(len(predictions)),
        "n_residual_rows": int(len(predictions)),
        "n_target_dates": int(predictions["target_date"].nunique()) if "target_date" in predictions.columns and not predictions.empty else 0,
    }


def _official_minus_model_implied_series(predictions: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    columns = ["date", "model_type"]
    for target in target_columns:
        columns.extend([f"official_{target}", f"model_implied_{target}", f"official_minus_model_implied_{target}"])
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(predictions["date"]).dt.date.astype(str),
            "model_type": predictions["model_type"].astype(str),
        }
    )
    for target in target_columns:
        official = predictions[f"actual_{target}"].astype(float)
        implied = predictions[f"pred_{target}"].astype(float)
        rows[f"official_{target}"] = official
        rows[f"model_implied_{target}"] = implied
        rows[f"official_minus_model_implied_{target}"] = official - implied
    return rows[columns]
