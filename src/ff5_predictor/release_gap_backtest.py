from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ff5_predictor.availability import make_release_gap_splits
from ff5_predictor.data_famafrench import load_ff5
from ff5_predictor.data_yfinance import load_market_data
from ff5_predictor.evaluation import evaluate_on_shared_dates, evaluate_prediction_groups, rank_models
from ff5_predictor.io import ensure_dir, normalize_datetime_index
from ff5_predictor.nowcast_features import build_nowcast_features
from ff5_predictor.nowcast_io import create_nowcast_run_dir, write_json, write_nowcast_predictions, write_yaml
from ff5_predictor.nowcast_models import ewma_predict, fit_ridge_nowcast, fit_tft_nowcast, rolling_mean_predict


@dataclass(frozen=True)
class ReleaseGapBacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    gap_metrics: dict[str, Any]
    run_dir: Path


def run_release_gap_backtest(config: dict[str, Any]) -> ReleaseGapBacktestResult:
    ff5_df = load_ff5(config)
    market_df = load_market_data(config)
    return run_release_gap_backtest_from_frames(ff5_df, market_df, config)


def run_release_gap_backtest_from_frames(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: dict[str, Any],
) -> ReleaseGapBacktestResult:
    ff5 = normalize_datetime_index(ff5_df)
    market = normalize_datetime_index(market_df)
    target_columns = list(config["prediction"]["target_columns"])
    aligned_dates = pd.DatetimeIndex(ff5.index.intersection(market.index)).sort_values()
    min_rows = int(config.get("nowcast", {}).get("min_train_rows", config.get("training", {}).get("min_train_rows", 1000)))
    step_rows = int(config.get("nowcast", {}).get("backtest_step_rows", 21))
    gap_sizes = [int(v) for v in config.get("availability", {}).get("release_gap_backtest_days", [1, 2, 3, 5, 10])]
    splits = make_release_gap_splits(aligned_dates, gap_sizes, min_rows, step_rows)
    grouped: dict[pd.Timestamp, list] = {}
    for split in splits:
        grouped.setdefault(split.cutoff_date, []).append(split)

    records: list[dict[str, Any]] = []
    for cutoff_date, cutoff_splits in grouped.items():
        max_gap = max(split.gap_size for split in cutoff_splits)
        target_dates = pd.DatetimeIndex(aligned_dates[(aligned_dates > cutoff_date)]).sort_values()[:max_gap]
        records.extend(_predict_gap_for_cutoff(ff5, market, cutoff_date, target_dates, cutoff_splits, config))

    predictions = pd.DataFrame(records)
    run_dir = create_nowcast_run_dir(config)
    write_yaml(run_dir / "config_resolved.yaml", config)
    write_nowcast_predictions(run_dir, "release_gap_predictions.csv", predictions)
    metrics = evaluate_prediction_groups(predictions, target_columns) if not predictions.empty else {}
    shared = evaluate_on_shared_dates(predictions, target_columns, baseline_model="ewma") if not predictions.empty else {}
    ranking = rank_models(shared)
    ensure_dir(run_dir / "metrics")
    ranking.to_csv(run_dir / "metrics" / "model_ranking.csv", index=False)
    gap_metrics = _gap_metrics(predictions, target_columns)
    write_json(run_dir / "metrics" / "metrics.json", metrics)
    write_json(run_dir / "metrics" / "shared_date_metrics.json", shared)
    write_json(run_dir / "metrics" / "gap_metrics.json", gap_metrics)
    return ReleaseGapBacktestResult(predictions=predictions, metrics=metrics, gap_metrics=gap_metrics, run_dir=run_dir)


def _predict_gap_for_cutoff(
    ff5: pd.DataFrame,
    market: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    target_dates: pd.DatetimeIndex,
    cutoff_splits: list,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    target_columns = list(config["prediction"]["target_columns"])
    models = list(config.get("nowcast", {}).get("models", ["rolling_mean", "ewma", "ridge"]))
    recursive = bool(config.get("availability", {}).get("recursive_factor_lags", True))
    train_df, feature_columns = _training_frame(ff5, market, cutoff_date, config)
    min_rows = int(config.get("nowcast", {}).get("min_train_rows", config.get("training", {}).get("min_train_rows", 1000)))
    if len(train_df) < min_rows:
        return []
    records: list[dict[str, Any]] = []
    train_window = int(config.get("nowcast", {}).get("train_window_rows", 2520))
    span = int(config.get("models", {}).get("ewma", {}).get("default_span", 21))
    fitted_ridge = fit_ridge_nowcast(train_df, feature_columns, target_columns, config) if "ridge" in models else None
    fitted_tft = fit_tft_nowcast(train_df, feature_columns, target_columns, config) if "tft" in models else None
    predictions_by_model: dict[str, pd.DataFrame] = {model: pd.DataFrame() for model in models}
    history_by_model = {model: ff5[target_columns].loc[:cutoff_date].copy() for model in models}
    feature_history_by_model = {model: train_df[feature_columns].copy() for model in models}

    split_by_size = {split.gap_size: set(pd.Timestamp(date) for date in split.target_dates) for split in cutoff_splits}
    for gap_day, target_date in enumerate(target_dates, start=1):
        actual = ff5.loc[target_date, target_columns]
        for model_type in models:
            if model_type == "ridge":
                if fitted_ridge is None:
                    continue
                recursive_predictions = predictions_by_model[model_type] if recursive else None
                feature_result = build_nowcast_features(
                    ff5.loc[:cutoff_date],
                    market.loc[:target_date],
                    config,
                    official_cutoff_date=cutoff_date,
                    recursive_predictions=recursive_predictions,
                )
                row = feature_result.features.reindex([target_date])
                for col in feature_columns:
                    if col not in row.columns:
                        row[col] = np.nan
                row = row[feature_columns]
                if row.isna().any(axis=None):
                    continue
                pred_values = fitted_ridge.predict_frame(row)[0]
                metadata = fitted_ridge.metadata
            elif model_type == "rolling_mean":
                pred_values = rolling_mean_predict(history_by_model[model_type], target_columns, train_window).to_numpy()
                metadata = _history_metadata(history_by_model[model_type])
            elif model_type == "ewma":
                pred_values = ewma_predict(history_by_model[model_type], target_columns, span).to_numpy()
                metadata = {**_history_metadata(history_by_model[model_type]), "span": span}
            elif model_type == "tft":
                if fitted_tft is None:
                    continue
                recursive_predictions = predictions_by_model[model_type] if recursive else None
                feature_result = build_nowcast_features(
                    ff5.loc[:cutoff_date],
                    market.loc[:target_date],
                    config,
                    official_cutoff_date=cutoff_date,
                    recursive_predictions=recursive_predictions,
                )
                row = feature_result.features.reindex([target_date])
                for col in feature_columns:
                    if col not in row.columns:
                        row[col] = np.nan
                row = row[feature_columns]
                if row.isna().any(axis=None):
                    continue
                feature_history = pd.concat([feature_history_by_model[model_type], row])
                pred_values = fitted_tft.predict_from_history(feature_history)
                metadata = fitted_tft.metadata
            else:
                continue

            pred_frame = pd.DataFrame([dict(zip(target_columns, pred_values))], index=[target_date])
            if recursive:
                predictions_by_model[model_type] = pd.concat([predictions_by_model[model_type], pred_frame])
                history_by_model[model_type] = pd.concat([history_by_model[model_type], pred_frame])
            if model_type == "tft":
                feature_history_by_model[model_type] = pd.concat([feature_history_by_model[model_type], row])

            for release_gap_size, dates in split_by_size.items():
                if target_date not in dates:
                    continue
                record = {
                    "date": target_date.date().isoformat(),
                    "cutoff_date": cutoff_date.date().isoformat(),
                    "target_date": target_date.date().isoformat(),
                    "gap_day": int(gap_day),
                    "release_gap_size": int(release_gap_size),
                    "model_type": model_type,
                    "train_start_date": metadata.get("train_start_date"),
                    "train_end_date": metadata.get("train_end_date"),
                    "n_train_rows": metadata.get("n_train_rows"),
                    "market_data_asof": target_date.date().isoformat(),
                    "factor_data_asof": cutoff_date.date().isoformat(),
                    "recursive_factor_lags": recursive,
                }
                for column, value in zip(target_columns, pred_values):
                    record[f"pred_{column}"] = float(value)
                    record[f"actual_{column}"] = float(actual[column])
                records.append(record)
    return records


def _training_frame(
    ff5: pd.DataFrame,
    market: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    target_columns = list(config["prediction"]["target_columns"])
    feature_result = build_nowcast_features(ff5.loc[:cutoff_date], market.loc[:cutoff_date], config)
    train_df = feature_result.features.join(ff5[target_columns].loc[:cutoff_date], how="inner")
    feature_columns = list(feature_result.feature_columns)
    train_df = train_df.dropna(subset=feature_columns + target_columns)
    return train_df, feature_columns


def _history_metadata(history: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_train_rows": int(len(history)),
        "train_start_date": str(pd.Timestamp(history.index.min()).date()),
        "train_end_date": str(pd.Timestamp(history.index.max()).date()),
    }


def _gap_metrics(predictions: pd.DataFrame, target_columns: list[str]) -> dict[str, Any]:
    if predictions.empty:
        return {"metrics_by_gap_day": {}, "metrics_by_release_gap_size": {}}
    return {
        "metrics_by_gap_day": {
            str(gap_day): evaluate_prediction_groups(group, target_columns)
            for gap_day, group in predictions.groupby("gap_day")
        },
        "metrics_by_release_gap_size": {
            str(size): evaluate_prediction_groups(group, target_columns)
            for size, group in predictions.groupby("release_gap_size")
        },
    }
