from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(
    predictions: pd.DataFrame,
    target_columns: list[str],
) -> dict[str, Any]:
    if predictions.empty:
        return {
            "model_type": None,
            "n_predictions": 0,
            "date_start": None,
            "date_end": None,
            "metrics_by_factor": {},
        }

    metrics_by_factor: dict[str, dict[str, float | None]] = {}
    for column in target_columns:
        pred = predictions[f"pred_{column}"].astype(float)
        actual = predictions[f"actual_{column}"].astype(float)
        corr = _safe_series_corr(pred, actual)
        metrics_by_factor[column] = {
            "mae": float(mean_absolute_error(actual, pred)),
            "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
            "r2": float(r2_score(actual, pred)) if len(actual) >= 2 else None,
            "directional_accuracy": float((np.sign(pred) == np.sign(actual)).mean()),
            "correlation": None if pd.isna(corr) else float(corr),
            "mean_prediction": float(pred.mean()),
            "prediction_std": float(pred.std(ddof=0)),
            "actual_std": float(actual.std(ddof=0)),
            "sign_bias": float((pred > 0).mean()),
            "top_quintile_hit_rate": _quantile_hit_rate(pred, actual, upper=True),
            "bottom_quintile_hit_rate": _quantile_hit_rate(pred, actual, upper=False),
        }

    dates = pd.to_datetime(predictions["date"])
    model_types = sorted(predictions["model_type"].dropna().unique().tolist())
    return {
        "model_type": model_types[0] if len(model_types) == 1 else model_types,
        "n_predictions": int(len(predictions)),
        "date_start": dates.min().date().isoformat(),
        "date_end": dates.max().date().isoformat(),
        "metrics_by_factor": metrics_by_factor,
    }


def compare_against_baseline(
    model_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    target_columns: list[str],
) -> dict[str, Any]:
    shared_dates = sorted(
        set(pd.to_datetime(model_predictions["date"])).intersection(
            set(pd.to_datetime(baseline_predictions["date"]))
        )
    )
    model_aligned = model_predictions[
        pd.to_datetime(model_predictions["date"]).isin(shared_dates)
    ]
    baseline_aligned = baseline_predictions[
        pd.to_datetime(baseline_predictions["date"]).isin(shared_dates)
    ]
    if not shared_dates:
        return {
            "n_shared_predictions": 0,
            "mae_delta_by_factor": {},
            "rmse_delta_by_factor": {},
        }
    model_metrics = evaluate_predictions(model_aligned, target_columns)["metrics_by_factor"]
    baseline_metrics = evaluate_predictions(baseline_aligned, target_columns)["metrics_by_factor"]
    comparison: dict[str, Any] = {
        "n_shared_predictions": len(shared_dates),
        "mae_delta_by_factor": {},
        "rmse_delta_by_factor": {},
    }
    for column in target_columns:
        comparison["mae_delta_by_factor"][column] = (
            model_metrics[column]["mae"] - baseline_metrics[column]["mae"]
        )
        comparison["rmse_delta_by_factor"][column] = (
            model_metrics[column]["rmse"] - baseline_metrics[column]["rmse"]
        )
    return comparison


def evaluate_prediction_groups(
    predictions: pd.DataFrame,
    target_columns: list[str],
    group_column: str = "model_type",
) -> dict[str, Any]:
    return {
        str(group): evaluate_predictions(group_df, target_columns)
        for group, group_df in predictions.groupby(group_column)
    }


def evaluate_on_shared_dates(
    predictions: pd.DataFrame,
    target_columns: list[str],
    group_column: str = "model_type",
    baseline_model: str = "rolling_mean",
) -> dict[str, Any]:
    if predictions.empty:
        return {"models": {}, "baseline_model": baseline_model, "n_shared_dates": 0}
    groups = {str(group): group_df.copy() for group, group_df in predictions.groupby(group_column)}
    date_sets = [set(pd.to_datetime(group_df["date"])) for group_df in groups.values()]
    shared_dates = sorted(set.intersection(*date_sets)) if date_sets else []
    shared_metrics = {
        model: evaluate_predictions(
            group_df[pd.to_datetime(group_df["date"]).isin(shared_dates)],
            target_columns,
        )
        for model, group_df in groups.items()
    }
    baseline_key = _resolve_baseline_key(shared_metrics, baseline_model)
    comparisons = {}
    if baseline_key is not None:
        baseline_df = groups[baseline_key][pd.to_datetime(groups[baseline_key]["date"]).isin(shared_dates)]
        for model, group_df in groups.items():
            if model == baseline_key:
                continue
            model_df = group_df[pd.to_datetime(group_df["date"]).isin(shared_dates)]
            comparisons[model] = compare_against_baseline(model_df, baseline_df, target_columns)
    return {
        "baseline_model": baseline_key,
        "n_shared_dates": len(shared_dates),
        "models": shared_metrics,
        "comparisons_vs_baseline": comparisons,
    }


def rank_models(
    shared_metrics: dict[str, Any],
    primary_metric: str = "rmse",
) -> pd.DataFrame:
    models = shared_metrics.get("models", shared_metrics)
    baseline_key = shared_metrics.get("baseline_model")
    rows = []
    for model, metrics in models.items():
        factor_metrics = metrics.get("metrics_by_factor", {})
        if not factor_metrics:
            continue
        values = list(factor_metrics.values())
        avg_mae = float(np.mean([m["mae"] for m in values]))
        avg_rmse = float(np.mean([m["rmse"] for m in values]))
        r2_values = [m["r2"] for m in values if m["r2"] is not None]
        avg_r2 = float(np.mean(r2_values)) if r2_values else np.nan
        avg_corr = float(np.mean([m["correlation"] or 0.0 for m in values]))
        avg_dir = float(np.mean([m["directional_accuracy"] for m in values]))
        rows.append(
            {
                "model_type": model,
                "avg_mae": avg_mae,
                "avg_rmse": avg_rmse,
                "avg_r2": avg_r2,
                "avg_corr": avg_corr,
                "avg_directional_accuracy": avg_dir,
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    if baseline_key in set(ranking["model_type"]):
        baseline_rmse = float(ranking.loc[ranking["model_type"] == baseline_key, "avg_rmse"].iloc[0])
        ranking["rmse_vs_baseline_pct"] = (ranking["avg_rmse"] / baseline_rmse - 1.0) * 100.0
    else:
        ranking["rmse_vs_baseline_pct"] = np.nan
    sort_column = f"avg_{primary_metric}"
    if sort_column not in ranking.columns:
        sort_column = "avg_rmse"
    return ranking.sort_values(sort_column).reset_index(drop=True)


def _resolve_baseline_key(metrics: dict[str, Any], baseline_model: str) -> str | None:
    if baseline_model in metrics:
        return baseline_model
    for key in metrics:
        if key.startswith(baseline_model):
            return key
    return next(iter(metrics), None) if metrics else None


def _quantile_hit_rate(pred: pd.Series, actual: pd.Series, upper: bool) -> float | None:
    if len(pred) < 5:
        return None
    threshold = pred.quantile(0.8 if upper else 0.2)
    mask = pred >= threshold if upper else pred <= threshold
    if not mask.any():
        return None
    actual_threshold = actual.quantile(0.8 if upper else 0.2)
    actual_hit = actual >= actual_threshold if upper else actual <= actual_threshold
    return float(actual_hit[mask].mean())


def _safe_series_corr(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if mask.sum() < 2:
        return None
    left_centered = left_values[mask] - left_values[mask].mean()
    right_centered = right_values[mask] - right_values[mask].mean()
    left_norm = float(np.sqrt(np.sum(left_centered * left_centered)))
    right_norm = float(np.sqrt(np.sum(right_centered * right_centered)))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(np.sum(left_centered * right_centered) / (left_norm * right_norm))
