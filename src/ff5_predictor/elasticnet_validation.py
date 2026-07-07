from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ff5_predictor.availability import filter_dates_by_config
from ff5_predictor.data_famafrench import load_ff5
from ff5_predictor.data_yfinance import load_market_data
from ff5_predictor.io import ensure_dir, normalize_datetime_index
from ff5_predictor.nowcast_features import build_nowcast_features
from ff5_predictor.nowcast_io import create_nowcast_run_dir, write_json, write_yaml
from ff5_predictor.nowcast_models import (
    FittedNowcastModel,
    FittedPerFactorNowcastModel,
    fit_elasticnet_nowcast,
    fit_per_factor_elasticnet_nowcast,
)


@dataclass(frozen=True)
class TimeSeriesValidationFold:
    protocol: str
    fold_id: int
    train_positions: np.ndarray
    validation_positions: np.ndarray
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp
    validation_end_date: pd.Timestamp


@dataclass(frozen=True)
class ElasticNetValidationResult:
    run_dir: Path
    fold_predictions: pd.DataFrame
    vintage_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    coefficient_table: pd.DataFrame
    coefficient_stability: pd.DataFrame
    vintage_stability: pd.DataFrame
    model_summary: pd.DataFrame
    metadata: dict[str, Any]


def run_elasticnet_validation(config: dict[str, Any]) -> ElasticNetValidationResult:
    ff5_df = load_ff5(config)
    market_df = load_market_data(config)
    return run_elasticnet_validation_from_frames(ff5_df, market_df, config)


def run_elasticnet_validation_from_frames(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: dict[str, Any],
) -> ElasticNetValidationResult:
    _validate_research_config(config)
    ff5 = normalize_datetime_index(ff5_df)
    market = normalize_datetime_index(market_df)
    model_df, feature_columns, target_columns = _build_validation_model_frame(ff5, market, config)
    folds = make_time_series_validation_folds(model_df.index, config)
    models = _validation_models(config)

    fold_prediction_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []
    for fold in folds:
        train_df = model_df.iloc[fold.train_positions]
        validation_df = model_df.iloc[fold.validation_positions]
        for model_type in models:
            fitted = _fit_validation_model(model_type, train_df, feature_columns, target_columns, config)
            pred = _predict_validation_frame(fitted, validation_df[feature_columns])
            fold_prediction_parts.append(
                _prediction_records(
                    validation_df,
                    pred,
                    target_columns,
                    model_type=model_type,
                    protocol=fold.protocol,
                    fold_id=fold.fold_id,
                    train_start_date=fold.train_start_date,
                    train_end_date=fold.train_end_date,
                    validation_start_date=fold.validation_start_date,
                    validation_end_date=fold.validation_end_date,
                    n_train_rows=len(train_df),
                    n_validation_rows=len(validation_df),
                    extra={},
                )
            )
            coefficient_parts.append(
                extract_elasticnet_coefficients(
                    fitted,
                    protocol=fold.protocol,
                    model_type=model_type,
                    fold_id=fold.fold_id,
                    train_start_date=fold.train_start_date,
                    train_end_date=fold.train_end_date,
                    validation_start_date=fold.validation_start_date,
                    validation_end_date=fold.validation_end_date,
                    n_train_rows=len(train_df),
                )
            )

    fold_predictions = (
        pd.concat(fold_prediction_parts, ignore_index=True) if fold_prediction_parts else pd.DataFrame()
    )
    coefficient_table = (
        pd.concat(coefficient_parts, ignore_index=True) if coefficient_parts else _empty_coefficient_table()
    )
    fold_metrics = _fold_metrics(fold_predictions, target_columns)
    coefficient_stability = _coefficient_stability(coefficient_table, config)

    vintage_predictions, vintage_coefficients = _run_vintage_holdout(
        model_df,
        feature_columns,
        target_columns,
        config,
        models,
    )
    vintage_stability = _vintage_stability(vintage_predictions, vintage_coefficients, target_columns, config)
    model_summary = _model_summary(fold_metrics)

    run_dir = create_nowcast_run_dir(config)
    _write_outputs(
        run_dir,
        config,
        model_df,
        feature_columns,
        target_columns,
        folds,
        fold_predictions,
        vintage_predictions,
        fold_metrics,
        coefficient_table,
        coefficient_stability,
        vintage_stability,
        model_summary,
    )
    metadata = _metadata(config, model_df, feature_columns, target_columns, folds)
    write_json(run_dir / "metadata" / "validation_metadata.json", metadata)
    return ElasticNetValidationResult(
        run_dir=run_dir,
        fold_predictions=fold_predictions,
        vintage_predictions=vintage_predictions,
        fold_metrics=fold_metrics,
        coefficient_table=coefficient_table,
        coefficient_stability=coefficient_stability,
        vintage_stability=vintage_stability,
        model_summary=model_summary,
        metadata=metadata,
    )


def make_time_series_validation_folds(
    index: pd.DatetimeIndex,
    config: dict[str, Any],
) -> list[TimeSeriesValidationFold]:
    dates = pd.DatetimeIndex(pd.to_datetime(index).tz_localize(None)).sort_values()
    validation_cfg = config.get("elasticnet_validation", {})
    protocols = [str(v) for v in validation_cfg.get("protocols", ["sliding", "expanding"])]
    validation_rows = int(validation_cfg.get("validation_window_rows", 252))
    fold_step_rows = int(validation_cfg.get("fold_step_rows", validation_rows))
    validation_start = pd.Timestamp(validation_cfg.get("validation_start_date", "2020-01-01"))
    train_window_rows = int(validation_cfg.get("train_window_rows", config.get("nowcast", {}).get("train_window_rows", 2520)))
    min_train_rows = int(validation_cfg.get("min_train_rows", config.get("nowcast", {}).get("min_train_rows", 1000)))
    if validation_rows < 1:
        raise ValueError("elasticnet_validation.validation_window_rows must be at least 1")
    if fold_step_rows < 1:
        raise ValueError("elasticnet_validation.fold_step_rows must be at least 1")
    if train_window_rows < 1:
        raise ValueError("elasticnet_validation.train_window_rows must be at least 1 for sliding folds")

    first_start = int(np.searchsorted(dates.values, validation_start.to_datetime64(), side="left"))
    folds: list[TimeSeriesValidationFold] = []
    fold_ids = {protocol: 0 for protocol in protocols}
    for start in range(first_start, len(dates) - validation_rows + 1, fold_step_rows):
        validation_positions = np.arange(start, start + validation_rows, dtype=int)
        for protocol in protocols:
            history_positions = np.arange(0, start, dtype=int)
            if protocol == "sliding":
                train_positions = history_positions[-train_window_rows:]
            elif protocol == "expanding":
                train_positions = history_positions
            else:
                raise ValueError(f"Unsupported elasticnet validation protocol: {protocol}")
            if len(train_positions) < min_train_rows:
                continue
            fold_id = fold_ids[protocol]
            fold_ids[protocol] += 1
            folds.append(
                TimeSeriesValidationFold(
                    protocol=protocol,
                    fold_id=fold_id,
                    train_positions=train_positions,
                    validation_positions=validation_positions,
                    train_start_date=pd.Timestamp(dates[train_positions[0]]),
                    train_end_date=pd.Timestamp(dates[train_positions[-1]]),
                    validation_start_date=pd.Timestamp(dates[validation_positions[0]]),
                    validation_end_date=pd.Timestamp(dates[validation_positions[-1]]),
                )
            )
    return folds


def extract_elasticnet_coefficients(
    fitted: FittedNowcastModel | FittedPerFactorNowcastModel,
    *,
    protocol: str,
    model_type: str,
    fold_id: int,
    train_start_date: pd.Timestamp,
    train_end_date: pd.Timestamp,
    validation_start_date: pd.Timestamp | None,
    validation_end_date: pd.Timestamp | None,
    n_train_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if isinstance(fitted, FittedPerFactorNowcastModel):
        for target in fitted.target_columns:
            submodel = fitted.fitted_by_target[target]
            coefficients = np.asarray(submodel.model.coef_, dtype=float).reshape(-1)
            intercept = float(np.asarray(submodel.model.intercept_).reshape(-1)[0])
            metadata = submodel.metadata
            for feature, coefficient in zip(submodel.feature_columns, coefficients):
                rows.append(
                    _coefficient_record(
                        protocol,
                        model_type,
                        fold_id,
                        target,
                        feature,
                        coefficient,
                        intercept,
                        metadata,
                        train_start_date,
                        train_end_date,
                        validation_start_date,
                        validation_end_date,
                        n_train_rows,
                    )
                )
    else:
        coefficients = np.asarray(fitted.model.coef_, dtype=float)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        intercepts = np.asarray(fitted.model.intercept_, dtype=float).reshape(-1)
        for target_idx, target in enumerate(fitted.target_columns):
            intercept = float(intercepts[target_idx] if target_idx < len(intercepts) else intercepts[0])
            for feature, coefficient in zip(fitted.feature_columns, coefficients[target_idx]):
                rows.append(
                    _coefficient_record(
                        protocol,
                        model_type,
                        fold_id,
                        target,
                        feature,
                        float(coefficient),
                        intercept,
                        fitted.metadata,
                        train_start_date,
                        train_end_date,
                        validation_start_date,
                        validation_end_date,
                        n_train_rows,
                    )
                )
    return pd.DataFrame(rows)


def _coefficient_record(
    protocol: str,
    model_type: str,
    fold_id: int,
    target: str,
    feature: str,
    coefficient: float,
    intercept: float,
    metadata: dict[str, Any],
    train_start_date: pd.Timestamp,
    train_end_date: pd.Timestamp,
    validation_start_date: pd.Timestamp | None,
    validation_end_date: pd.Timestamp | None,
    n_train_rows: int,
) -> dict[str, Any]:
    threshold = 0.0
    return {
        "protocol": protocol,
        "model_type": model_type,
        "fold_id": int(fold_id),
        "target": target,
        "feature": feature,
        "coefficient": float(coefficient),
        "abs_coefficient": float(abs(coefficient)),
        "intercept": float(intercept),
        "nonzero": bool(abs(coefficient) > threshold),
        "alpha": float(metadata.get("alpha", np.nan)),
        "l1_ratio": float(metadata.get("l1_ratio", np.nan)),
        "tune_alpha": False,
        "coefficient_space": "standardized_feature_space" if metadata.get("scale_features", True) else "raw_feature_space",
        "scale_features": bool(metadata.get("scale_features", True)),
        "train_start_date": pd.Timestamp(train_start_date).date().isoformat(),
        "train_end_date": pd.Timestamp(train_end_date).date().isoformat(),
        "validation_start_date": pd.Timestamp(validation_start_date).date().isoformat()
        if validation_start_date is not None
        else None,
        "validation_end_date": pd.Timestamp(validation_end_date).date().isoformat()
        if validation_end_date is not None
        else None,
        "n_train_rows": int(n_train_rows),
    }


def _build_validation_model_frame(
    ff5: pd.DataFrame,
    market: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    target_columns = list(config["prediction"]["target_columns"])
    feature_result = build_nowcast_features(ff5.iloc[0:0], market, config)
    feature_frame = feature_result.features.dropna(axis=1, how="all")
    missing_targets = [target for target in target_columns if target not in ff5.columns]
    if missing_targets:
        raise ValueError(f"Official factor data is missing configured targets: {missing_targets}")
    model_df = feature_frame.join(ff5[target_columns], how="inner")
    model_df = model_df.loc[filter_dates_by_config(model_df.index, config)]
    feature_columns = list(feature_frame.columns)
    model_df = model_df.dropna(subset=feature_columns + target_columns)
    if model_df.empty:
        raise ValueError("ElasticNet validation modeling dataset is empty")
    return model_df, feature_columns, target_columns


def _validate_research_config(config: dict[str, Any]) -> None:
    if bool(config.get("target_features", {}).get("include_lagged_targets", False)):
        raise ValueError("ElasticNet validation requires target_features.include_lagged_targets=false")
    if bool(config.get("availability", {}).get("recursive_factor_lags", False)):
        raise ValueError("ElasticNet validation requires availability.recursive_factor_lags=false")
    if bool(config.get("feature_extraction", {}).get("enabled", False)):
        raise ValueError("ElasticNet validation does not use feature extraction")
    unsupported = set(config.get("nowcast", {}).get("models", [])) - {"elasticnet", "per_factor_elasticnet"}
    if unsupported:
        raise ValueError(f"ElasticNet validation only supports elasticnet/per_factor_elasticnet, got {sorted(unsupported)}")


def _validation_models(config: dict[str, Any]) -> list[str]:
    models = list(config.get("nowcast", {}).get("models", ["elasticnet", "per_factor_elasticnet"]))
    return [model for model in models if model in {"elasticnet", "per_factor_elasticnet"}]


def _fit_validation_model(
    model_type: str,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict[str, Any],
) -> FittedNowcastModel | FittedPerFactorNowcastModel:
    fit_config = _fixed_elasticnet_config(config, model_type)
    if model_type == "elasticnet":
        return fit_elasticnet_nowcast(train_df, feature_columns, target_columns, fit_config)
    if model_type == "per_factor_elasticnet":
        return fit_per_factor_elasticnet_nowcast(train_df, feature_columns, target_columns, fit_config)
    raise ValueError(f"Unsupported validation model: {model_type}")


def _fixed_elasticnet_config(config: dict[str, Any], model_type: str) -> dict[str, Any]:
    result = deepcopy(config)
    validation_cfg = result.get("elasticnet_validation", {})
    fixed = validation_cfg.get("fixed_hyperparameters", {})
    model_cfg = {
        **result.get("models", {}).get("elasticnet", {}),
        **result.get("models", {}).get(model_type, {}),
        **fixed,
        "tune_alpha": False,
        "tune_l1_ratio": False,
    }
    model_cfg.setdefault("alpha", 0.001)
    model_cfg.setdefault("l1_ratio", 0.05)
    model_cfg.setdefault("max_iter", 50000)
    model_cfg.setdefault("tol", 0.0001)
    model_cfg.setdefault("scale_features", True)
    result.setdefault("models", {})[model_type] = model_cfg
    result.setdefault("models", {})["elasticnet"] = {**result.get("models", {}).get("elasticnet", {}), **model_cfg}
    result.setdefault("nowcast", {})["models"] = [model_type]
    result["nowcast"]["train_window_rows"] = 0
    result["feature_extraction"] = {"enabled": False, "method": "none", "apply_to_models": []}
    return result


def _predict_validation_frame(
    fitted: FittedNowcastModel | FittedPerFactorNowcastModel,
    features: pd.DataFrame,
) -> np.ndarray:
    if isinstance(fitted, FittedPerFactorNowcastModel):
        columns = []
        for target in fitted.target_columns:
            submodel = fitted.fitted_by_target[target]
            pred = np.asarray(submodel.predict_frame(features), dtype=float).reshape(-1)
            columns.append(pred)
        return np.column_stack(columns)
    pred = np.asarray(fitted.predict_frame(features), dtype=float)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    return pred


def _prediction_records(
    validation_df: pd.DataFrame,
    pred: np.ndarray,
    target_columns: list[str],
    *,
    model_type: str,
    protocol: str,
    fold_id: int | None,
    train_start_date: pd.Timestamp,
    train_end_date: pd.Timestamp,
    validation_start_date: pd.Timestamp,
    validation_end_date: pd.Timestamp,
    n_train_rows: int,
    n_validation_rows: int,
    extra: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_idx, date in enumerate(validation_df.index):
        record: dict[str, Any] = {
            "date": pd.Timestamp(date).date().isoformat(),
            "target_date": pd.Timestamp(date).date().isoformat(),
            "protocol": protocol,
            "fold_id": fold_id,
            "model_type": model_type,
            "train_start_date": pd.Timestamp(train_start_date).date().isoformat(),
            "train_end_date": pd.Timestamp(train_end_date).date().isoformat(),
            "validation_start_date": pd.Timestamp(validation_start_date).date().isoformat(),
            "validation_end_date": pd.Timestamp(validation_end_date).date().isoformat(),
            "n_train_rows": int(n_train_rows),
            "n_validation_rows": int(n_validation_rows),
            **extra,
        }
        for target_idx, target in enumerate(target_columns):
            actual = float(validation_df.iloc[row_idx][target])
            predicted = float(pred[row_idx, target_idx])
            error = predicted - actual
            record[f"pred_{target}"] = predicted
            record[f"actual_{target}"] = actual
            record[f"error_{target}"] = error
            record[f"abs_error_{target}"] = abs(error)
        rows.append(record)
    return pd.DataFrame(rows)


def _fold_metrics(predictions: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["protocol", "model_type", "fold_id"]
    for keys, group in predictions.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        for target in target_columns:
            pred = group[f"pred_{target}"].astype(float).to_numpy()
            actual = group[f"actual_{target}"].astype(float).to_numpy()
            error = pred - actual
            rows.append(
                {
                    **base,
                    "target": target,
                    "n": int(len(group)),
                    "validation_start_date": group["validation_start_date"].iloc[0],
                    "validation_end_date": group["validation_end_date"].iloc[0],
                    "rmse": _rmse(error),
                    "mae": float(np.mean(np.abs(error))),
                    "mean_error": float(np.mean(error)),
                    "correlation": _safe_corr(pred, actual),
                    "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(actual))),
                }
            )
    return pd.DataFrame(rows)


def _coefficient_stability(coefficient_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if coefficient_table.empty:
        return pd.DataFrame()
    threshold = float(config.get("elasticnet_validation", {}).get("nonzero_threshold", 1e-10))
    top_n = int(config.get("elasticnet_validation", {}).get("top_n_features", 50))
    rows: list[dict[str, Any]] = []
    for (protocol, model_type, target), group in coefficient_table.groupby(["protocol", "model_type", "target"]):
        fold_ids = sorted(group["fold_id"].dropna().unique())
        for previous_fold, current_fold in zip(fold_ids[:-1], fold_ids[1:]):
            previous = group[group["fold_id"] == previous_fold].set_index("feature")["coefficient"].astype(float)
            current = group[group["fold_id"] == current_fold].set_index("feature")["coefficient"].astype(float)
            aligned = pd.concat([previous.rename("previous"), current.rename("current")], axis=1).fillna(0.0)
            previous_values = aligned["previous"].to_numpy()
            current_values = aligned["current"].to_numpy()
            previous_nonzero = set(aligned.index[np.abs(previous_values) > threshold])
            current_nonzero = set(aligned.index[np.abs(current_values) > threshold])
            union = previous_nonzero | current_nonzero
            top_count = min(top_n, len(aligned))
            previous_top = set(aligned["previous"].abs().nlargest(top_count).index)
            current_top = set(aligned["current"].abs().nlargest(top_count).index)
            rows.append(
                {
                    "protocol": protocol,
                    "model_type": model_type,
                    "target": target,
                    "previous_fold_id": int(previous_fold),
                    "current_fold_id": int(current_fold),
                    "coefficient_correlation": _safe_corr(previous_values, current_values),
                    "sign_agreement": float(np.mean(np.sign(previous_values) == np.sign(current_values))),
                    "top_50_feature_overlap": float(len(previous_top & current_top) / top_count) if top_count else np.nan,
                    "nonzero_set_jaccard": float(len(previous_nonzero & current_nonzero) / len(union)) if union else 1.0,
                    "normalized_l2_drift": _normalized_l2_drift(previous_values, current_values),
                    "n_features": int(len(aligned)),
                    "n_nonzero_previous": int(len(previous_nonzero)),
                    "n_nonzero_current": int(len(current_nonzero)),
                }
            )
    return pd.DataFrame(rows)


def _run_vintage_holdout(
    model_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict[str, Any],
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_cfg = config.get("elasticnet_validation", {})
    holdout_rows = int(validation_cfg.get("holdout_rows", 252))
    vintage_step_rows = int(validation_cfg.get("vintage_step_rows", 252))
    max_vintages = int(validation_cfg.get("max_vintages", 6))
    train_window_rows = int(validation_cfg.get("train_window_rows", config.get("nowcast", {}).get("train_window_rows", 2520)))
    min_train_rows = int(validation_cfg.get("min_train_rows", config.get("nowcast", {}).get("min_train_rows", 1000)))
    protocols = [str(v) for v in validation_cfg.get("protocols", ["sliding", "expanding"])]
    if len(model_df) <= holdout_rows + min_train_rows:
        return pd.DataFrame(), pd.DataFrame()
    holdout_positions = np.arange(len(model_df) - holdout_rows, len(model_df), dtype=int)
    holdout_df = model_df.iloc[holdout_positions]
    holdout_start_pos = int(holdout_positions[0])
    max_age_count = max(0, (holdout_start_pos - min_train_rows) // vintage_step_rows)
    age_rows = [i * vintage_step_rows for i in range(min(max_vintages, max_age_count + 1))]
    prediction_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []
    for protocol in protocols:
        for vintage_id, staleness_rows in enumerate(age_rows):
            cutoff_pos = holdout_start_pos - 1 - staleness_rows
            if cutoff_pos < min_train_rows - 1:
                continue
            history_positions = np.arange(0, cutoff_pos + 1, dtype=int)
            if protocol == "sliding":
                train_positions = history_positions[-train_window_rows:]
            elif protocol == "expanding":
                train_positions = history_positions
            else:
                continue
            if len(train_positions) < min_train_rows:
                continue
            train_df = model_df.iloc[train_positions]
            for model_type in models:
                fitted = _fit_validation_model(model_type, train_df, feature_columns, target_columns, config)
                pred = _predict_validation_frame(fitted, holdout_df[feature_columns])
                prediction_parts.append(
                    _prediction_records(
                        holdout_df,
                        pred,
                        target_columns,
                        model_type=model_type,
                        protocol=protocol,
                        fold_id=None,
                        train_start_date=pd.Timestamp(train_df.index[0]),
                        train_end_date=pd.Timestamp(train_df.index[-1]),
                        validation_start_date=pd.Timestamp(holdout_df.index[0]),
                        validation_end_date=pd.Timestamp(holdout_df.index[-1]),
                        n_train_rows=len(train_df),
                        n_validation_rows=len(holdout_df),
                        extra={
                            "vintage_id": int(vintage_id),
                            "staleness_rows": int(staleness_rows),
                            "holdout_start_date": pd.Timestamp(holdout_df.index[0]).date().isoformat(),
                            "holdout_end_date": pd.Timestamp(holdout_df.index[-1]).date().isoformat(),
                        },
                    )
                )
                coef = extract_elasticnet_coefficients(
                    fitted,
                    protocol=protocol,
                    model_type=model_type,
                    fold_id=vintage_id,
                    train_start_date=pd.Timestamp(train_df.index[0]),
                    train_end_date=pd.Timestamp(train_df.index[-1]),
                    validation_start_date=pd.Timestamp(holdout_df.index[0]),
                    validation_end_date=pd.Timestamp(holdout_df.index[-1]),
                    n_train_rows=len(train_df),
                )
                coef["vintage_id"] = int(vintage_id)
                coef["staleness_rows"] = int(staleness_rows)
                coefficient_parts.append(coef)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    coefficients = pd.concat(coefficient_parts, ignore_index=True) if coefficient_parts else _empty_coefficient_table()
    return predictions, coefficients


def _vintage_stability(
    vintage_predictions: pd.DataFrame,
    vintage_coefficients: pd.DataFrame,
    target_columns: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    if vintage_predictions.empty:
        return pd.DataFrame()
    metric_rows: list[dict[str, Any]] = []
    drift_lookup = _prediction_drift_vs_freshest(vintage_predictions, target_columns)
    coefficient_lookup = _coefficient_drift_vs_freshest(vintage_coefficients, config)
    group_cols = ["protocol", "model_type", "vintage_id", "staleness_rows"]
    for keys, group in vintage_predictions.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        for target in target_columns:
            pred = group[f"pred_{target}"].astype(float).to_numpy()
            actual = group[f"actual_{target}"].astype(float).to_numpy()
            error = pred - actual
            lookup_key = (base["protocol"], base["model_type"], int(base["vintage_id"]), target)
            metric_rows.append(
                {
                    **base,
                    "target": target,
                    "n": int(len(group)),
                    "rmse": _rmse(error),
                    "mae": float(np.mean(np.abs(error))),
                    "correlation": _safe_corr(pred, actual),
                    "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(actual))),
                    **drift_lookup.get(lookup_key, {}),
                    **coefficient_lookup.get(lookup_key, {}),
                }
            )
    return pd.DataFrame(metric_rows)


def _prediction_drift_vs_freshest(
    vintage_predictions: pd.DataFrame,
    target_columns: list[str],
) -> dict[tuple[str, str, int, str], dict[str, float]]:
    results: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for (protocol, model_type), group in vintage_predictions.groupby(["protocol", "model_type"]):
        freshest_id = int(group["staleness_rows"].astype(int).idxmin())
        freshest_staleness = int(vintage_predictions.loc[freshest_id, "staleness_rows"])
        freshest = group[group["staleness_rows"].astype(int) == freshest_staleness]
        for target in target_columns:
            fresh_series = freshest.set_index("target_date")[f"pred_{target}"].astype(float)
            for (vintage_id, staleness_rows), vintage_group in group.groupby(["vintage_id", "staleness_rows"]):
                current = vintage_group.set_index("target_date")[f"pred_{target}"].astype(float)
                aligned = pd.concat([current.rename("current"), fresh_series.rename("freshest")], axis=1).dropna()
                if aligned.empty:
                    continue
                drift = aligned["current"] - aligned["freshest"]
                results[(protocol, model_type, int(vintage_id), target)] = {
                    "prediction_drift_mae_vs_freshest": float(drift.abs().mean()),
                    "prediction_drift_rmse_vs_freshest": _rmse(drift.to_numpy()),
                }
    return results


def _coefficient_drift_vs_freshest(
    vintage_coefficients: pd.DataFrame,
    config: dict[str, Any],
) -> dict[tuple[str, str, int, str], dict[str, float]]:
    if vintage_coefficients.empty:
        return {}
    top_n = int(config.get("elasticnet_validation", {}).get("top_n_features", 50))
    results: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for (protocol, model_type, target), group in vintage_coefficients.groupby(["protocol", "model_type", "target"]):
        freshest_staleness = int(group["staleness_rows"].astype(int).min())
        freshest = group[group["staleness_rows"].astype(int) == freshest_staleness].set_index("feature")["coefficient"].astype(float)
        for (vintage_id, staleness_rows), vintage_group in group.groupby(["vintage_id", "staleness_rows"]):
            current = vintage_group.set_index("feature")["coefficient"].astype(float)
            aligned = pd.concat([current.rename("current"), freshest.rename("freshest")], axis=1).fillna(0.0)
            top_count = min(top_n, len(aligned))
            current_top = set(aligned["current"].abs().nlargest(top_count).index)
            freshest_top = set(aligned["freshest"].abs().nlargest(top_count).index)
            results[(protocol, model_type, int(vintage_id), target)] = {
                "coefficient_correlation_vs_freshest": _safe_corr(
                    aligned["current"].to_numpy(),
                    aligned["freshest"].to_numpy(),
                ),
                "coefficient_l2_drift_vs_freshest": _normalized_l2_drift(
                    aligned["freshest"].to_numpy(),
                    aligned["current"].to_numpy(),
                ),
                "top_50_overlap_vs_freshest": float(len(current_top & freshest_top) / top_count) if top_count else np.nan,
            }
    return results


def _model_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (protocol, model_type, target), group in fold_metrics.groupby(["protocol", "model_type", "target"]):
        rows.append(
            {
                "protocol": protocol,
                "model_type": model_type,
                "target": target,
                "n_folds": int(group["fold_id"].nunique()),
                "avg_rmse": float(group["rmse"].mean()),
                "avg_mae": float(group["mae"].mean()),
                "avg_corr": float(group["correlation"].mean()),
                "avg_directional_accuracy": float(group["directional_accuracy"].mean()),
            }
        )
    for (protocol, model_type), group in fold_metrics.groupby(["protocol", "model_type"]):
        rows.append(
            {
                "protocol": protocol,
                "model_type": model_type,
                "target": "__average__",
                "n_folds": int(group["fold_id"].nunique()),
                "avg_rmse": float(group["rmse"].mean()),
                "avg_mae": float(group["mae"].mean()),
                "avg_corr": float(group["correlation"].mean()),
                "avg_directional_accuracy": float(group["directional_accuracy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_outputs(
    run_dir: Path,
    config: dict[str, Any],
    model_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    folds: list[TimeSeriesValidationFold],
    fold_predictions: pd.DataFrame,
    vintage_predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    coefficient_stability: pd.DataFrame,
    vintage_stability: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    write_yaml(run_dir / "config_resolved.yaml", config)
    ensure_dir(run_dir / "predictions")
    ensure_dir(run_dir / "tables")
    ensure_dir(run_dir / "figures")
    fold_predictions.to_csv(run_dir / "predictions" / "fold_predictions.csv", index=False)
    vintage_predictions.to_csv(run_dir / "predictions" / "vintage_holdout_predictions.csv", index=False)
    fold_metrics.to_csv(run_dir / "tables" / "fold_metrics.csv", index=False)
    coefficient_table.to_csv(run_dir / "tables" / "coefficient_table.csv", index=False)
    coefficient_stability.to_csv(run_dir / "tables" / "coefficient_stability.csv", index=False)
    vintage_stability.to_csv(run_dir / "tables" / "vintage_stability.csv", index=False)
    model_summary.to_csv(run_dir / "tables" / "model_summary.csv", index=False)
    figure_paths = _write_figures(run_dir, fold_metrics, coefficient_table, coefficient_stability, vintage_stability)
    _write_report(
        run_dir,
        config,
        model_df,
        feature_columns,
        target_columns,
        folds,
        model_summary,
        coefficient_stability,
        vintage_stability,
        figure_paths,
    )


def _write_figures(
    run_dir: Path,
    fold_metrics: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    coefficient_stability: pd.DataFrame,
    vintage_stability: pd.DataFrame,
) -> dict[str, str]:
    figures_dir = ensure_dir(run_dir / "figures")
    paths: dict[str, str] = {}
    if not fold_metrics.empty:
        avg_rmse = (
            fold_metrics.groupby(["protocol", "model_type", "fold_id", "validation_end_date"], as_index=False)["rmse"].mean()
        )
        paths["fold_rmse_over_time"] = _write_line_svg(
            figures_dir / "fold_rmse_over_time.svg",
            avg_rmse,
            x_col="fold_id",
            y_col="rmse",
            series_cols=["protocol", "model_type"],
            title="Fold RMSE over time",
            y_label="RMSE",
        )
    else:
        paths["fold_rmse_over_time"] = _write_placeholder_svg(figures_dir / "fold_rmse_over_time.svg", "No fold metrics")
    if not coefficient_stability.empty:
        stability = coefficient_stability.groupby(["protocol", "model_type", "current_fold_id"], as_index=False)[
            "coefficient_correlation"
        ].mean()
        paths["coefficient_stability"] = _write_line_svg(
            figures_dir / "coefficient_stability_over_folds.svg",
            stability,
            x_col="current_fold_id",
            y_col="coefficient_correlation",
            series_cols=["protocol", "model_type"],
            title="Adjacent-fold coefficient correlation",
            y_label="Correlation",
        )
    else:
        paths["coefficient_stability"] = _write_placeholder_svg(
            figures_dir / "coefficient_stability_over_folds.svg",
            "No coefficient stability metrics",
        )
    paths["top_feature_paths"] = _write_top_feature_paths_svg(figures_dir / "top_feature_coefficient_paths.svg", coefficient_table)
    if not vintage_stability.empty:
        vintage_rmse = vintage_stability.groupby(["protocol", "model_type", "staleness_rows"], as_index=False)["rmse"].mean()
        paths["vintage_holdout_rmse"] = _write_line_svg(
            figures_dir / "vintage_holdout_rmse_vs_staleness.svg",
            vintage_rmse,
            x_col="staleness_rows",
            y_col="rmse",
            series_cols=["protocol", "model_type"],
            title="Holdout RMSE by model staleness",
            y_label="RMSE",
        )
    else:
        paths["vintage_holdout_rmse"] = _write_placeholder_svg(
            figures_dir / "vintage_holdout_rmse_vs_staleness.svg",
            "No vintage holdout metrics",
        )
    return {key: str(Path(value).relative_to(run_dir)) for key, value in paths.items()}


def _write_line_svg(
    path: Path,
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    series_cols: list[str],
    title: str,
    y_label: str,
) -> str:
    width, height = 920, 420
    margin_left, margin_right, margin_top, margin_bottom = 72, 26, 48, 64
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    clean = data.dropna(subset=[x_col, y_col]).copy()
    if clean.empty:
        return _write_placeholder_svg(path, f"No data for {title}")
    x_values = clean[x_col].astype(float).to_numpy()
    y_values = clean[y_col].astype(float).to_numpy()
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= abs(y_min) * 0.1 + 1e-6
        y_max += abs(y_max) * 0.1 + 1e-6
    colors = ["#174a5b", "#b66b2d", "#6f7f3f", "#5e4b8b", "#a33f4f", "#2f6c8f"]
    lines = []
    legends = []
    for idx, (series_key, group) in enumerate(clean.groupby(series_cols)):
        label = " / ".join(str(v) for v in (series_key if isinstance(series_key, tuple) else (series_key,)))
        color = colors[idx % len(colors)]
        points = []
        for _, row in group.sort_values(x_col).iterrows():
            x = margin_left + (float(row[x_col]) - x_min) / (x_max - x_min) * plot_w
            y = margin_top + (1.0 - (float(row[y_col]) - y_min) / (y_max - y_min)) * plot_h
            points.append(f"{x:.2f},{y:.2f}")
        if points:
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
            legend_y = height - margin_bottom + 28 + 20 * idx
            legends.append(
                f'<line x1="{margin_left}" y1="{legend_y}" x2="{margin_left + 24}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="2.4"/><text x="{margin_left + 32}" y="{legend_y + 5}" '
                f'font-size="13">{escape(label)}</text>'
            )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fffaf0"/>
<text x="{margin_left}" y="28" font-size="20" font-weight="700" fill="#1d2b36">{escape(title)}</text>
<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#97a1a8"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#97a1a8"/>
<text x="12" y="{margin_top + 14}" font-size="13" fill="#1d2b36">{escape(y_label)}</text>
<text x="{margin_left}" y="{margin_top + plot_h + 22}" font-size="12" fill="#1d2b36">{x_min:g}</text>
<text x="{margin_left + plot_w - 40}" y="{margin_top + plot_h + 22}" font-size="12" fill="#1d2b36">{x_max:g}</text>
<text x="18" y="{margin_top + 6}" font-size="12" fill="#1d2b36">{y_max:.4g}</text>
<text x="18" y="{margin_top + plot_h}" font-size="12" fill="#1d2b36">{y_min:.4g}</text>
{''.join(lines)}
{''.join(legends)}
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)


def _write_top_feature_paths_svg(path: Path, coefficient_table: pd.DataFrame) -> str:
    if coefficient_table.empty:
        return _write_placeholder_svg(path, "No coefficient data")
    candidates = coefficient_table[
        (coefficient_table["protocol"] == "sliding") & (coefficient_table["model_type"] == "elasticnet")
    ]
    if candidates.empty:
        candidates = coefficient_table.copy()
    target = str(candidates["target"].iloc[0])
    target_rows = candidates[candidates["target"] == target]
    top_features = (
        target_rows.groupby("feature")["abs_coefficient"].mean().sort_values(ascending=False).head(8).index.tolist()
    )
    data = target_rows[target_rows["feature"].isin(top_features)].copy()
    if data.empty:
        return _write_placeholder_svg(path, "No top feature paths")
    data["series"] = data["feature"]
    return _write_line_svg(
        path,
        data,
        x_col="fold_id",
        y_col="coefficient",
        series_cols=["series"],
        title=f"Top coefficient paths for {target}",
        y_label="Coefficient",
    )


def _write_placeholder_svg(path: Path, message: str) -> str:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="920" height="260" viewBox="0 0 920 260">
<rect width="100%" height="100%" fill="#fffaf0"/>
<text x="40" y="130" font-size="20" fill="#1d2b36">{escape(message)}</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)


def _write_report(
    run_dir: Path,
    config: dict[str, Any],
    model_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    folds: list[TimeSeriesValidationFold],
    model_summary: pd.DataFrame,
    coefficient_stability: pd.DataFrame,
    vintage_stability: pd.DataFrame,
    figure_paths: dict[str, str],
) -> None:
    summary_html = model_summary.round(6).to_html(index=False) if not model_summary.empty else "<p>No model summary.</p>"
    coef_summary = (
        coefficient_stability.groupby(["protocol", "model_type", "target"], as_index=False)
        .agg(
            coefficient_correlation=("coefficient_correlation", "mean"),
            sign_agreement=("sign_agreement", "mean"),
            top_50_feature_overlap=("top_50_feature_overlap", "mean"),
            nonzero_set_jaccard=("nonzero_set_jaccard", "mean"),
            normalized_l2_drift=("normalized_l2_drift", "mean"),
        )
        .round(4)
        .to_html(index=False)
        if not coefficient_stability.empty
        else "<p>No coefficient stability rows.</p>"
    )
    vintage_summary = (
        vintage_stability.groupby(["protocol", "model_type", "staleness_rows"], as_index=False)
        .agg(
            rmse=("rmse", "mean"),
            prediction_drift_mae_vs_freshest=("prediction_drift_mae_vs_freshest", "mean"),
            coefficient_l2_drift_vs_freshest=("coefficient_l2_drift_vs_freshest", "mean"),
        )
        .round(6)
        .to_html(index=False)
        if not vintage_stability.empty
        else "<p>No vintage stability rows.</p>"
    )
    figures = "\n".join(
        f'<section><img src="{escape(path)}" alt="{escape(name)}"/></section>' for name, path in figure_paths.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>ElasticNet Time-Series Validation</title>
  <style>
    body {{ font-family: Inter, system-ui, sans-serif; margin: 32px; color: #1d2b36; background: #fffaf0; }}
    h1, h2 {{ color: #174a5b; }}
    table {{ border-collapse: collapse; margin: 16px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8cab5; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #d8cab5; background: white; margin: 12px 0 24px; }}
    .meta {{ color: #65716e; }}
  </style>
</head>
<body>
  <h1>ElasticNet Time-Series Validation</h1>
  <p class="meta">Rows: {len(model_df):,}; features: {len(feature_columns):,}; targets: {escape(', '.join(target_columns))}; folds: {len(folds):,}.</p>
  <p>This report uses time-ordered sliding and expanding validation folds. Each extractor-free ElasticNet model is fit only on rows before the validation block, with fixed hyperparameters.</p>
  <h2>Model Summary</h2>
  {summary_html}
  <h2>Coefficient Stability Summary</h2>
  {coef_summary}
  <h2>Vintage Holdout Summary</h2>
  {vintage_summary}
  <h2>Figures</h2>
  {figures}
  <h2>Fixed Hyperparameters</h2>
  <pre>{escape(str(config.get("elasticnet_validation", {}).get("fixed_hyperparameters", {})))}</pre>
</body>
</html>"""
    (run_dir / "validation_report.html").write_text(html, encoding="utf-8")


def _metadata(
    config: dict[str, Any],
    model_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    folds: list[TimeSeriesValidationFold],
) -> dict[str, Any]:
    return {
        "methodology": "elasticnet_time_series_validation",
        "description": (
            "Sliding and expanding time-series validation for fixed-hyperparameter shared and per-factor ElasticNet. "
            "Official FF factors are labels only; feature extraction and target lags are disabled."
        ),
        "date_start": pd.Timestamp(model_df.index.min()).date().isoformat(),
        "date_end": pd.Timestamp(model_df.index.max()).date().isoformat(),
        "n_modeling_rows": int(len(model_df)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "n_folds": int(len(folds)),
        "fold_counts_by_protocol": pd.Series([fold.protocol for fold in folds]).value_counts().to_dict() if folds else {},
        "models": _validation_models(config),
        "fixed_hyperparameters": {
            **config.get("models", {}).get("elasticnet", {}),
            **config.get("elasticnet_validation", {}).get("fixed_hyperparameters", {}),
            "tune_alpha": False,
            "tune_l1_ratio": False,
        },
    }


def _empty_coefficient_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "protocol",
            "model_type",
            "fold_id",
            "target",
            "feature",
            "coefficient",
            "abs_coefficient",
            "intercept",
            "nonzero",
            "alpha",
            "l1_ratio",
            "tune_alpha",
            "coefficient_space",
            "scale_features",
            "train_start_date",
            "train_end_date",
            "validation_start_date",
            "validation_end_date",
            "n_train_rows",
        ]
    )


def _rmse(error: np.ndarray | pd.Series) -> float:
    values = np.asarray(error, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def _safe_corr(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _normalized_l2_drift(previous: np.ndarray, current: np.ndarray) -> float:
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    denominator = float(np.linalg.norm(previous))
    if denominator == 0.0:
        return float(np.linalg.norm(current))
    return float(np.linalg.norm(current - previous) / denominator)
