from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ff5_predictor.availability import unreleased_market_dates
from ff5_predictor.data_famafrench import load_ff5
from ff5_predictor.data_yfinance import load_market_data
from ff5_predictor.io import ensure_dir, normalize_datetime_index
from ff5_predictor.nowcast_dataset import build_nowcast_dataset
from ff5_predictor.nowcast_features import build_nowcast_features
from ff5_predictor.nowcast_io import (
    create_nowcast_run_dir,
    sync_latest_copy,
    write_json,
    write_nowcast_predictions,
    write_yaml,
)
from ff5_predictor.nowcast_models import (
    FittedNowcastModel,
    ewma_predict,
    fit_ridge_nowcast,
    fit_tft_nowcast,
    rolling_mean_predict,
    save_fitted_model,
)


@dataclass(frozen=True)
class ProductionNowcastResult:
    predictions: pd.DataFrame
    feature_snapshot: pd.DataFrame
    metadata: dict[str, Any]
    run_dir: Path


def run_production_nowcast(config: dict[str, Any]) -> ProductionNowcastResult:
    ff5_df = load_ff5(config)
    market_df = load_market_data(config)
    return run_production_nowcast_from_frames(ff5_df, market_df, config)


def run_production_nowcast_from_frames(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: dict[str, Any],
) -> ProductionNowcastResult:
    run_dir = create_nowcast_run_dir(config)
    write_yaml(run_dir / "config_resolved.yaml", config)
    dataset = build_nowcast_dataset(ff5_df, market_df, config)
    write_json(run_dir / "metadata" / "dataset_metadata.json", dataset.metadata)

    predictions = _empty_prediction_frame(dataset.target_columns)
    feature_snapshot = pd.DataFrame()
    if len(dataset.unreleased_dates) > 0:
        predictions, feature_snapshot = _predict_unreleased(ff5_df, market_df, dataset, config, run_dir)

    write_nowcast_predictions(run_dir, "latest_nowcast.csv", predictions)
    write_json(run_dir / "predictions" / "latest_nowcast.json", {"records": predictions.to_dict(orient="records")})
    if bool(config.get("nowcast", {}).get("save_feature_snapshot", True)):
        ensure_dir(run_dir / "features")
        feature_snapshot.to_parquet(run_dir / "features" / "latest_feature_snapshot.parquet")

    metadata = {
        **dataset.metadata,
        "n_prediction_rows": int(len(predictions)),
        "models": config.get("nowcast", {}).get("models", []),
        "primary_model": config.get("nowcast", {}).get("primary_model"),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "metadata" / "metadata.json", metadata)
    if config.get("nowcast", {}).get("run_name") == "production_latest":
        sync_latest_copy(config, run_dir)
    return ProductionNowcastResult(
        predictions=predictions,
        feature_snapshot=feature_snapshot,
        metadata=metadata,
        run_dir=run_dir,
    )


def _predict_unreleased(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
    dataset,
    config: dict[str, Any],
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ff5 = normalize_datetime_index(ff5_df)
    market = normalize_datetime_index(market_df)
    target_columns = dataset.target_columns
    models = list(config.get("nowcast", {}).get("models", ["ridge"]))
    recursive = bool(config.get("availability", {}).get("recursive_factor_lags", True))
    records: list[dict[str, Any]] = []
    snapshots: list[pd.DataFrame] = []

    fitted_ridge: FittedNowcastModel | None = None
    if "ridge" in models:
        fitted_ridge = fit_ridge_nowcast(dataset.train_df, dataset.feature_columns, target_columns, config)
        if bool(config.get("nowcast", {}).get("save_model_artifact", True)):
            save_primary_model_artifact(fitted_ridge, run_dir)
    fitted_tft = fit_tft_nowcast(dataset.train_df, dataset.feature_columns, target_columns, config) if "tft" in models else None

    history_by_model = {
        model: ff5[target_columns].loc[: dataset.latest_official_date].copy()
        for model in models
    }
    feature_history_by_model = {
        model: dataset.train_df[dataset.feature_columns].copy()
        for model in models
    }
    recursive_predictions_by_model: dict[str, pd.DataFrame] = {model: pd.DataFrame() for model in models}
    span = int(config.get("models", {}).get("ewma", {}).get("default_span", 21))
    train_window = int(config.get("nowcast", {}).get("train_window_rows", 2520))

    for gap_day, target_date in enumerate(dataset.unreleased_dates, start=1):
        for model_type in models:
            if model_type == "ridge":
                if fitted_ridge is None:
                    continue
                recursive_predictions = recursive_predictions_by_model[model_type] if recursive else None
                feature_result = build_nowcast_features(
                    ff5,
                    market.loc[:target_date],
                    config,
                    official_cutoff_date=dataset.latest_official_date,
                    recursive_predictions=recursive_predictions,
                )
                row = feature_result.features.reindex([target_date])
                missing = [col for col in dataset.feature_columns if col not in row.columns]
                for col in missing:
                    row[col] = np.nan
                row = row[dataset.feature_columns]
                if row.isna().any(axis=None):
                    continue
                pred_values = fitted_ridge.predict_frame(row)[0]
                snapshot = row.copy()
                snapshot["model_type"] = model_type
                snapshots.append(snapshot)
                model_metadata = fitted_ridge.metadata
            elif model_type == "rolling_mean":
                pred_values = rolling_mean_predict(history_by_model[model_type], target_columns, train_window).to_numpy()
                model_metadata = {
                    "n_train_rows": int(len(history_by_model[model_type])),
                    "train_start_date": str(history_by_model[model_type].index.min().date()),
                    "train_end_date": str(history_by_model[model_type].index.max().date()),
                }
            elif model_type == "ewma":
                pred_values = ewma_predict(history_by_model[model_type], target_columns, span).to_numpy()
                model_metadata = {
                    "n_train_rows": int(len(history_by_model[model_type])),
                    "train_start_date": str(history_by_model[model_type].index.min().date()),
                    "train_end_date": str(history_by_model[model_type].index.max().date()),
                    "span": span,
                }
            elif model_type == "tft":
                if fitted_tft is None:
                    continue
                recursive_predictions = recursive_predictions_by_model[model_type] if recursive else None
                feature_result = build_nowcast_features(
                    ff5,
                    market.loc[:target_date],
                    config,
                    official_cutoff_date=dataset.latest_official_date,
                    recursive_predictions=recursive_predictions,
                )
                row = feature_result.features.reindex([target_date])
                missing = [col for col in dataset.feature_columns if col not in row.columns]
                for col in missing:
                    row[col] = np.nan
                row = row[dataset.feature_columns]
                if row.isna().any(axis=None):
                    continue
                feature_history = pd.concat([feature_history_by_model[model_type], row])
                pred_values = fitted_tft.predict_from_history(feature_history)
                model_metadata = fitted_tft.metadata
            else:
                raise ValueError(f"Unsupported production nowcast model: {model_type}")

            record = _prediction_record(
                target_date=target_date,
                pred_values=pred_values,
                target_columns=target_columns,
                model_type=model_type,
                latest_official_date=dataset.latest_official_date,
                latest_market_date=dataset.latest_market_date,
                gap_day=gap_day,
                model_metadata=model_metadata,
                recursive=recursive,
            )
            records.append(record)
            pred_frame = pd.DataFrame([dict(zip(target_columns, pred_values))], index=[target_date])
            if recursive:
                recursive_predictions_by_model[model_type] = pd.concat(
                    [recursive_predictions_by_model[model_type], pred_frame]
                )
                history_by_model[model_type] = pd.concat([history_by_model[model_type], pred_frame])
            if model_type == "tft":
                feature_history_by_model[model_type] = pd.concat([feature_history_by_model[model_type], row])

    predictions = pd.DataFrame(records, columns=_prediction_columns(target_columns))
    feature_snapshot = pd.concat(snapshots) if snapshots else pd.DataFrame()
    return predictions, feature_snapshot


def save_primary_model_artifact(
    fitted: FittedNowcastModel,
    run_dir: Path,
) -> None:
    ensure_dir(run_dir / "models")
    save_fitted_model(fitted, run_dir / "models" / f"{fitted.model_type}.joblib")


def _prediction_record(
    target_date: pd.Timestamp,
    pred_values,
    target_columns: list[str],
    model_type: str,
    latest_official_date: pd.Timestamp,
    latest_market_date: pd.Timestamp,
    gap_day: int,
    model_metadata: dict[str, Any],
    recursive: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "date": pd.Timestamp(target_date).date().isoformat(),
        "model_type": model_type,
        "latest_official_factor_date": pd.Timestamp(latest_official_date).date().isoformat(),
        "latest_market_date": pd.Timestamp(latest_market_date).date().isoformat(),
        "is_unreleased": True,
        "gap_day": int(gap_day),
        "train_start_date": model_metadata.get("train_start_date"),
        "train_end_date": model_metadata.get("train_end_date"),
        "n_train_rows": model_metadata.get("n_train_rows"),
        "market_data_asof": pd.Timestamp(target_date).date().isoformat(),
        "factor_data_asof": pd.Timestamp(latest_official_date).date().isoformat(),
        "recursive_factor_lags": recursive,
    }
    for column, value in zip(target_columns, pred_values):
        record[f"pred_{column}"] = float(value)
    return record


def _prediction_columns(target_columns: list[str]) -> list[str]:
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


def _empty_prediction_frame(target_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=_prediction_columns(target_columns))
