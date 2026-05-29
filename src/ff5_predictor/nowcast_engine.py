from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ff5_predictor.io import normalize_datetime_index
from ff5_predictor.nowcast_features import build_nowcast_features
from ff5_predictor.nowcast_models import (
    ewma_predict,
    fit_ridge_nowcast,
    fit_tft_nowcast,
    rolling_mean_predict,
)


@dataclass(frozen=True)
class NowcastTargetSpec:
    target_dates: pd.DatetimeIndex
    cutoff_date: pd.Timestamp
    latest_market_date: pd.Timestamp
    actuals: pd.DataFrame | None
    is_unreleased: bool
    release_gap_size_by_date: dict[pd.Timestamp, list[int]] | None = None


@dataclass
class NowcastEngineResult:
    predictions: pd.DataFrame
    feature_snapshots: pd.DataFrame
    fitted_models: dict[str, Any]
    metadata: dict[str, Any]


def run_nowcast_engine(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    spec: NowcastTargetSpec,
    config: dict[str, Any],
) -> NowcastEngineResult:
    ff5 = normalize_datetime_index(ff5_df)
    market = normalize_datetime_index(market_df)
    target_dates = pd.DatetimeIndex(pd.to_datetime(spec.target_dates).tz_localize(None)).sort_values()
    models = list(config.get("nowcast", {}).get("models", ["ridge"]))
    recursive = bool(config.get("availability", {}).get("recursive_factor_lags", True))
    train_window = int(config.get("nowcast", {}).get("train_window_rows", 2520))
    span = int(config.get("models", {}).get("ewma", {}).get("default_span", 21))

    fitted_models: dict[str, Any] = {}
    if "ridge" in models and not train_df.empty:
        fitted_models["ridge"] = fit_ridge_nowcast(train_df, feature_columns, target_columns, config)
    if "tft" in models and not train_df.empty:
        fitted_models["tft"] = fit_tft_nowcast(train_df, feature_columns, target_columns, config)

    predictions_by_model: dict[str, pd.DataFrame] = {model: pd.DataFrame() for model in models}
    history_by_model = {model: ff5[target_columns].loc[: spec.cutoff_date].copy() for model in models}
    feature_history_by_model = {model: train_df[feature_columns].copy() for model in models}
    records: list[dict[str, Any]] = []
    snapshots: list[pd.DataFrame] = []

    for gap_day, target_date in enumerate(target_dates, start=1):
        target_date = pd.Timestamp(target_date)
        for model_type in models:
            pred_values: np.ndarray | None = None
            model_metadata: dict[str, Any]
            row: pd.DataFrame | None = None

            if model_type == "ridge":
                fitted = fitted_models.get("ridge")
                if fitted is None:
                    continue
                row = _feature_row(
                    ff5,
                    market,
                    target_date,
                    spec.cutoff_date,
                    config,
                    feature_columns,
                    predictions_by_model[model_type] if recursive else None,
                )
                if row is None:
                    continue
                pred_values = fitted.predict_frame(row)[0]
                model_metadata = fitted.metadata
                snapshot = row.copy()
                snapshot["date"] = target_date.date().isoformat()
                snapshot["model_type"] = model_type
                snapshots.append(snapshot)
            elif model_type == "rolling_mean":
                history = history_by_model[model_type]
                if history.empty:
                    continue
                pred_values = rolling_mean_predict(history, target_columns, train_window).to_numpy()
                model_metadata = _history_metadata(history)
            elif model_type == "ewma":
                history = history_by_model[model_type]
                if history.empty:
                    continue
                pred_values = ewma_predict(history, target_columns, span).to_numpy()
                model_metadata = {**_history_metadata(history), "span": span}
            elif model_type == "tft":
                fitted = fitted_models.get("tft")
                if fitted is None:
                    continue
                row = _feature_row(
                    ff5,
                    market,
                    target_date,
                    spec.cutoff_date,
                    config,
                    feature_columns,
                    predictions_by_model[model_type] if recursive else None,
                )
                if row is None:
                    continue
                feature_history = pd.concat([feature_history_by_model[model_type], row])
                pred_values = fitted.predict_from_history(feature_history)
                model_metadata = fitted.metadata
            else:
                continue

            if pred_values is None:
                continue
            record = _prediction_record(
                target_date=target_date,
                pred_values=pred_values,
                target_columns=target_columns,
                model_type=model_type,
                cutoff_date=spec.cutoff_date,
                latest_market_date=spec.latest_market_date,
                gap_day=gap_day,
                model_metadata=model_metadata,
                recursive=recursive,
                is_unreleased=spec.is_unreleased,
                actuals=spec.actuals,
            )
            release_gap_sizes = _release_gap_sizes(spec, target_date)
            if release_gap_sizes:
                for release_gap_size in release_gap_sizes:
                    records.append({**record, "release_gap_size": int(release_gap_size)})
            else:
                records.append(record)

            pred_frame = pd.DataFrame([dict(zip(target_columns, pred_values))], index=[target_date])
            if recursive:
                predictions_by_model[model_type] = pd.concat([predictions_by_model[model_type], pred_frame])
                history_by_model[model_type] = pd.concat([history_by_model[model_type], pred_frame])
            if model_type == "tft" and row is not None:
                feature_history_by_model[model_type] = pd.concat([feature_history_by_model[model_type], row])

    predictions = pd.DataFrame(records)
    feature_snapshots = pd.concat(snapshots) if snapshots else pd.DataFrame()
    if not feature_snapshots.empty:
        feature_snapshots.index = pd.to_datetime(feature_snapshots["date"])
    return NowcastEngineResult(
        predictions=predictions,
        feature_snapshots=feature_snapshots,
        fitted_models=fitted_models,
        metadata={
            "models": models,
            "n_prediction_rows": int(len(predictions)),
            "n_feature_snapshot_rows": int(len(feature_snapshots)),
            "recursive_factor_lags": recursive,
            "cutoff_date": pd.Timestamp(spec.cutoff_date).date().isoformat(),
            "latest_market_date": pd.Timestamp(spec.latest_market_date).date().isoformat(),
        },
    )


def production_prediction_columns(target_columns: list[str]) -> list[str]:
    return [
        "date",
        *[f"pred_{column}" for column in target_columns],
        "model_type",
        "latest_official_factor_date",
        "latest_market_date",
        "is_unreleased",
        "gap_day",
        "train_start_date",
        "train_end_date",
        "n_train_rows",
        "market_data_asof",
        "factor_data_asof",
        "recursive_factor_lags",
    ]


def backtest_prediction_columns(target_columns: list[str]) -> list[str]:
    return [
        "date",
        "cutoff_date",
        "target_date",
        "gap_day",
        "release_gap_size",
        "model_type",
        "train_start_date",
        "train_end_date",
        "n_train_rows",
        "market_data_asof",
        "factor_data_asof",
        "recursive_factor_lags",
        *[f"pred_{column}" for column in target_columns],
        *[f"actual_{column}" for column in target_columns],
    ]


def empty_production_predictions(target_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=production_prediction_columns(target_columns))


def empty_backtest_predictions(target_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=backtest_prediction_columns(target_columns))


def select_production_columns(predictions: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    columns = production_prediction_columns(target_columns)
    if predictions.empty:
        return empty_production_predictions(target_columns)
    result = predictions.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    return result[columns]


def select_backtest_columns(predictions: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    columns = backtest_prediction_columns(target_columns)
    if predictions.empty:
        return empty_backtest_predictions(target_columns)
    result = predictions.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    return result[columns]


def _feature_row(
    ff5: pd.DataFrame,
    market: pd.DataFrame,
    target_date: pd.Timestamp,
    cutoff_date: pd.Timestamp,
    config: dict[str, Any],
    feature_columns: list[str],
    recursive_predictions: pd.DataFrame | None,
) -> pd.DataFrame | None:
    feature_result = build_nowcast_features(
        ff5.loc[:cutoff_date],
        market.loc[:target_date],
        config,
        official_cutoff_date=cutoff_date,
        recursive_predictions=recursive_predictions,
    )
    row = feature_result.features.reindex([target_date])
    for column in feature_columns:
        if column not in row.columns:
            row[column] = np.nan
    row = row[feature_columns]
    if row.isna().any(axis=None):
        return None
    return row


def _prediction_record(
    target_date: pd.Timestamp,
    pred_values,
    target_columns: list[str],
    model_type: str,
    cutoff_date: pd.Timestamp,
    latest_market_date: pd.Timestamp,
    gap_day: int,
    model_metadata: dict[str, Any],
    recursive: bool,
    is_unreleased: bool,
    actuals: pd.DataFrame | None,
) -> dict[str, Any]:
    target_date = pd.Timestamp(target_date)
    record: dict[str, Any] = {
        "date": target_date.date().isoformat(),
        "target_date": target_date.date().isoformat(),
        "model_type": model_type,
        "cutoff_date": pd.Timestamp(cutoff_date).date().isoformat(),
        "latest_official_factor_date": pd.Timestamp(cutoff_date).date().isoformat(),
        "latest_market_date": pd.Timestamp(latest_market_date).date().isoformat(),
        "is_unreleased": bool(is_unreleased),
        "gap_day": int(gap_day),
        "train_start_date": model_metadata.get("train_start_date"),
        "train_end_date": model_metadata.get("train_end_date"),
        "n_train_rows": model_metadata.get("n_train_rows"),
        "market_data_asof": target_date.date().isoformat(),
        "factor_data_asof": pd.Timestamp(cutoff_date).date().isoformat(),
        "recursive_factor_lags": recursive,
    }
    for column, value in zip(target_columns, pred_values):
        record[f"pred_{column}"] = float(value)
    if actuals is not None and target_date in actuals.index:
        actual = actuals.loc[target_date]
        for column in target_columns:
            record[f"actual_{column}"] = float(actual[column])
    return record


def _history_metadata(history: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_train_rows": int(len(history)),
        "train_start_date": str(pd.Timestamp(history.index.min()).date()),
        "train_end_date": str(pd.Timestamp(history.index.max()).date()),
    }


def _release_gap_sizes(spec: NowcastTargetSpec, target_date: pd.Timestamp) -> list[int] | None:
    if not spec.release_gap_size_by_date:
        return None
    sizes = spec.release_gap_size_by_date.get(pd.Timestamp(target_date), [])
    return [int(size) for size in sizes]
