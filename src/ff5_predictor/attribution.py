from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ff5_predictor.nowcast_models import FittedNowcastModel


SUPPORTED_LINEAR_ATTRIBUTION_MODELS = {"ridge", "elasticnet"}


@dataclass(frozen=True)
class AttributionResult:
    contribution_long: pd.DataFrame
    contribution_wide: pd.DataFrame
    top_contributions: pd.DataFrame
    coefficient_table: pd.DataFrame
    group_summary: pd.DataFrame
    metadata: dict[str, Any]


def explain_linear_predictions(
    fitted: FittedNowcastModel,
    feature_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    top_n: int,
    feature_groups: dict[str, list[str]] | None = None,
) -> AttributionResult:
    if fitted.model_type not in SUPPORTED_LINEAR_ATTRIBUTION_MODELS:
        supported = ", ".join(sorted(SUPPORTED_LINEAR_ATTRIBUTION_MODELS))
        raise ValueError(f"Linear attribution only supports fitted model_type in {{{supported}}}")
    if not hasattr(fitted.model, "coef_") or not hasattr(fitted.model, "intercept_"):
        raise ValueError("Fitted model does not expose linear coefficients and intercepts")

    top_n = max(int(top_n), 0)
    coefficients = np.asarray(fitted.model.coef_, dtype=float)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)
    intercepts = np.asarray(fitted.model.intercept_, dtype=float).reshape(-1)
    if coefficients.shape != (len(fitted.target_columns), len(fitted.feature_columns)):
        raise ValueError(
            "Coefficient shape does not match target and feature columns: "
            f"{coefficients.shape} vs ({len(fitted.target_columns)}, {len(fitted.feature_columns)})"
        )

    coefficient_table = _coefficient_table(fitted, coefficients)
    if feature_frame.empty or prediction_frame.empty:
        empty_long = _empty_long_frame()
        empty_top = _empty_top_frame()
        empty_group = _empty_group_frame()
        return AttributionResult(
            contribution_long=empty_long,
            contribution_wide=pd.DataFrame(columns=["date"]),
            top_contributions=empty_top,
            coefficient_table=coefficient_table,
            group_summary=empty_group,
            metadata={
                "model_type": fitted.model_type,
                "top_n": top_n,
                "n_prediction_rows": 0,
                "n_features": len(fitted.feature_columns),
                "n_targets": len(fitted.target_columns),
            },
        )

    features = _prepare_feature_frame(feature_frame, fitted.feature_columns)
    predictions = _prepare_prediction_frame(prediction_frame)
    common_dates = features.index.intersection(predictions.index)
    features = features.loc[common_dates]
    predictions = predictions.loc[common_dates]
    if features.empty:
        raise ValueError("No overlapping dates between feature_frame and prediction_frame")

    raw_frame = features[fitted.feature_columns].astype(float)
    raw_values = raw_frame.to_numpy()
    if fitted.scaler is not None:
        standardized = fitted.scaler.transform(raw_frame)
    else:
        standardized = raw_values.copy()

    records: list[dict[str, Any]] = []
    top_records: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    wide_parts: list[pd.DataFrame] = []
    group_lookup = _feature_group_lookup(fitted.feature_columns, feature_groups)

    for target_idx, target in enumerate(fitted.target_columns):
        target_coefs = coefficients[target_idx]
        contributions = standardized * target_coefs
        reconstructed = intercepts[target_idx] + contributions.sum(axis=1)
        pred_col = f"pred_{target}"
        if pred_col not in predictions.columns:
            raise ValueError(f"Prediction frame missing required column: {pred_col}")
        predicted = predictions[pred_col].astype(float).to_numpy()
        max_error = float(np.max(np.abs(reconstructed - predicted)))
        if max_error > 1e-8:
            raise ValueError(
                f"Linear attribution reconstruction failed for {target}: "
                f"max abs error {max_error:.3e}"
            )

        target_wide_columns: dict[str, np.ndarray] = {}
        for feature_idx, feature in enumerate(fitted.feature_columns):
            values = contributions[:, feature_idx]
            target_wide_columns[f"contrib_{target}__{feature}"] = values
            for row_idx, date in enumerate(features.index):
                records.append(
                    {
                        "date": pd.Timestamp(date).date().isoformat(),
                        "model_type": fitted.model_type,
                        "target": target,
                        "feature": feature,
                        "feature_value": float(raw_values[row_idx, feature_idx]),
                        "standardized_feature_value": float(standardized[row_idx, feature_idx]),
                        "coefficient": float(target_coefs[feature_idx]),
                        "contribution": float(values[row_idx]),
                        "intercept": float(intercepts[target_idx]),
                        "prediction": float(predicted[row_idx]),
                    }
                )
        wide_parts.append(pd.DataFrame(target_wide_columns, index=features.index))

        for row_idx, date in enumerate(features.index):
            order = np.argsort(np.abs(contributions[row_idx]))[::-1][:top_n]
            for rank, feature_idx in enumerate(order, start=1):
                contribution = float(contributions[row_idx, feature_idx])
                top_records.append(
                    {
                        "date": pd.Timestamp(date).date().isoformat(),
                        "model_type": fitted.model_type,
                        "target": target,
                        "rank": rank,
                        "feature": fitted.feature_columns[feature_idx],
                        "feature_value": float(raw_values[row_idx, feature_idx]),
                        "standardized_feature_value": float(standardized[row_idx, feature_idx]),
                        "coefficient": float(target_coefs[feature_idx]),
                        "contribution": contribution,
                        "abs_contribution": abs(contribution),
                        "prediction": float(predicted[row_idx]),
                    }
                )

            grouped: dict[str, float] = {}
            for feature_idx, feature in enumerate(fitted.feature_columns):
                group = group_lookup[feature]
                grouped[group] = grouped.get(group, 0.0) + float(contributions[row_idx, feature_idx])
            for group, value in sorted(grouped.items()):
                group_records.append(
                    {
                        "date": pd.Timestamp(date).date().isoformat(),
                        "model_type": fitted.model_type,
                        "target": target,
                        "feature_group": group,
                        "group_contribution": value,
                        "abs_group_contribution": abs(value),
                        "prediction": float(predicted[row_idx]),
                    }
                )

    contribution_wide = pd.concat(wide_parts, axis=1).reset_index(names="date")
    contribution_wide["date"] = pd.to_datetime(contribution_wide["date"]).dt.date.astype(str)
    contribution_long = pd.DataFrame.from_records(records, columns=_empty_long_frame().columns)
    top_contributions = pd.DataFrame.from_records(top_records, columns=_empty_top_frame().columns)
    group_summary = pd.DataFrame.from_records(group_records, columns=_empty_group_frame().columns)
    return AttributionResult(
        contribution_long=contribution_long,
        contribution_wide=contribution_wide,
        top_contributions=top_contributions,
        coefficient_table=coefficient_table,
        group_summary=group_summary,
        metadata={
            "model_type": fitted.model_type,
            "top_n": top_n,
            "n_prediction_rows": int(len(features)),
            "n_features": len(fitted.feature_columns),
            "n_targets": len(fitted.target_columns),
            "reconstruction_tolerance": 1e-8,
        },
    )


def explain_ridge_predictions(
    fitted: FittedNowcastModel,
    feature_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    top_n: int,
    feature_groups: dict[str, list[str]] | None = None,
) -> AttributionResult:
    if fitted.model_type != "ridge":
        raise ValueError("Ridge attribution only supports fitted model_type='ridge'")
    return explain_linear_predictions(
        fitted,
        feature_frame,
        prediction_frame,
        top_n=top_n,
        feature_groups=feature_groups,
    )


def infer_feature_group(feature: str) -> str:
    if feature.startswith("proxy_size_"):
        return "proxy_size"
    if feature.startswith("proxy_value_") or feature.startswith("proxy_growth_"):
        return "proxy_value"
    if feature.startswith("proxy_sector_"):
        return "proxy_sector"
    if feature.startswith("proxy_credit_") or feature.startswith("proxy_vix_") or feature.startswith("proxy_tlt_"):
        return "proxy_risk"
    if feature.startswith("proxy_global_"):
        return "proxy_global"
    if feature.startswith("proxy_quality_"):
        return "proxy_quality"
    if feature.startswith("proxy_realestate_"):
        return "proxy_realestate"
    if feature.startswith("proxy_market_"):
        return "proxy_market"
    if feature.endswith("_log_ret_1d"):
        return "market_log_returns"
    if feature.endswith("_ret_1d"):
        return "market_returns"
    if (
        feature.endswith("_oc_ret")
        or feature.endswith("_hl_range")
        or feature.endswith("_gap")
        or "_hl_range_mean_" in feature
    ):
        return "ohlc_intraday"
    if "_vol_" in feature and feature.endswith("d"):
        return "rolling_volatility"
    if "_drawdown_" in feature and feature.endswith("d"):
        return "drawdown"
    if "_ret_" in feature and feature.endswith("d"):
        return "rolling_returns"
    return "other"


def _coefficient_table(fitted: FittedNowcastModel, coefficients: np.ndarray) -> pd.DataFrame:
    records = []
    for target_idx, target in enumerate(fitted.target_columns):
        for feature_idx, feature in enumerate(fitted.feature_columns):
            coefficient = float(coefficients[target_idx, feature_idx])
            records.append(
                {
                    "model_type": fitted.model_type,
                    "target": target,
                    "feature": feature,
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                    "selected_alpha": fitted.metadata.get("alpha"),
                    "scale_features": fitted.metadata.get("scale_features"),
                    "train_start_date": fitted.metadata.get("train_start_date"),
                    "train_end_date": fitted.metadata.get("train_end_date"),
                    "n_train_rows": fitted.metadata.get("n_train_rows"),
                }
            )
    return pd.DataFrame.from_records(records)


def _prepare_feature_frame(feature_frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = feature_frame.copy()
    if "date" in features.columns:
        features.index = pd.to_datetime(features["date"])
    features.index = pd.DatetimeIndex(pd.to_datetime(features.index)).tz_localize(None)
    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise ValueError(f"Feature frame missing required columns: {missing}")
    return features.sort_index()


def _prepare_prediction_frame(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    predictions = prediction_frame.copy()
    if "date" not in predictions.columns:
        raise ValueError("Prediction frame must contain a date column")
    predictions.index = pd.DatetimeIndex(pd.to_datetime(predictions["date"])).tz_localize(None)
    return predictions.sort_index()


def _feature_group_lookup(
    feature_columns: list[str],
    feature_groups: dict[str, list[str]] | None,
) -> dict[str, str]:
    explicit: dict[str, str] = {}
    for group, features in (feature_groups or {}).items():
        for feature in features:
            explicit[feature] = group
    return {feature: explicit.get(feature, infer_feature_group(feature)) for feature in feature_columns}


def _empty_long_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "model_type",
            "target",
            "feature",
            "feature_value",
            "standardized_feature_value",
            "coefficient",
            "contribution",
            "intercept",
            "prediction",
        ]
    )


def _empty_top_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "model_type",
            "target",
            "rank",
            "feature",
            "feature_value",
            "standardized_feature_value",
            "coefficient",
            "contribution",
            "abs_contribution",
            "prediction",
        ]
    )


def _empty_group_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "model_type",
            "target",
            "feature_group",
            "group_contribution",
            "abs_group_contribution",
            "prediction",
        ]
    )
