from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TargetTransformResult:
    modeling_df: pd.DataFrame
    target_columns: list[str]
    actual_columns: list[str]
    residual_columns: list[str]
    baseline_columns: list[str]
    original_target_columns: list[str]
    target_mode: str
    target_horizon_rows: int
    residual_baseline_type: str | None = None
    residual_baseline_window_rows: int | None = None


def apply_target_transform(
    modeling_df: pd.DataFrame,
    original_target_columns: list[str],
    config: dict[str, Any],
) -> TargetTransformResult:
    prediction = config.get("prediction", {})
    target_mode = str(prediction.get("target_mode", "daily"))
    horizon_rows = int(prediction.get("cumulative_horizon_rows", 1))
    df = modeling_df.sort_index().copy()

    base_target_columns = list(original_target_columns)
    if horizon_rows > 1:
        actual_columns = []
        for column in original_target_columns:
            transformed = f"{column}_fwd_{horizon_rows}d"
            df[transformed] = _forward_rolling_sum(df[column], horizon_rows)
            actual_columns.append(transformed)
    else:
        actual_columns = base_target_columns

    if target_mode == "daily" and horizon_rows == 1:
        return TargetTransformResult(
            modeling_df=df.dropna(subset=base_target_columns),
            target_columns=base_target_columns,
            actual_columns=base_target_columns,
            residual_columns=[],
            baseline_columns=[],
            original_target_columns=base_target_columns,
            target_mode=target_mode,
            target_horizon_rows=horizon_rows,
        )

    if target_mode == "cumulative":
        transformed_df = df.dropna(subset=actual_columns).copy()
        return TargetTransformResult(
            modeling_df=transformed_df,
            target_columns=actual_columns,
            actual_columns=actual_columns,
            residual_columns=[],
            baseline_columns=[],
            original_target_columns=base_target_columns,
            target_mode=target_mode,
            target_horizon_rows=horizon_rows,
        )

    if target_mode == "residual":
        residual_cfg = prediction.get("residual_baseline", {})
        baseline_type = str(residual_cfg.get("type", "rolling_mean"))
        if baseline_type != "rolling_mean":
            raise ValueError(f"Unsupported residual baseline type: {baseline_type}")
        window_rows = int(residual_cfg.get("window_rows", 1260))
        baseline_columns: list[str] = []
        residual_columns: list[str] = []
        for source_column, original_column in zip(actual_columns, original_target_columns):
            baseline_column = f"{original_column}_baseline"
            residual_column = f"{original_column}_residual"
            df[baseline_column] = df[source_column].shift(1).rolling(window_rows).mean()
            df[residual_column] = df[source_column] - df[baseline_column]
            baseline_columns.append(baseline_column)
            residual_columns.append(residual_column)
        transformed_df = df.dropna(subset=actual_columns + baseline_columns + residual_columns).copy()
        return TargetTransformResult(
            modeling_df=transformed_df,
            target_columns=residual_columns,
            actual_columns=actual_columns,
            residual_columns=residual_columns,
            baseline_columns=baseline_columns,
            original_target_columns=base_target_columns,
            target_mode=target_mode,
            target_horizon_rows=horizon_rows,
            residual_baseline_type=baseline_type,
            residual_baseline_window_rows=window_rows,
        )

    raise ValueError(f"Unsupported prediction.target_mode: {target_mode}")


def reconstruct_predictions(
    predictions: pd.DataFrame,
    transform_result: TargetTransformResult,
    config: dict[str, Any],
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    result = predictions.copy()
    lookup_columns = (
        transform_result.actual_columns
        + transform_result.baseline_columns
        + transform_result.residual_columns
    )
    if lookup_columns:
        lookup = transform_result.modeling_df[lookup_columns].copy()
        lookup.index = pd.to_datetime(lookup.index)
    else:
        lookup = pd.DataFrame(index=transform_result.modeling_df.index)
    result["date"] = pd.to_datetime(result["date"])
    result = result.merge(lookup, left_on="date", right_index=True, how="left", suffixes=("", "_lookup"))

    if transform_result.target_mode == "residual":
        for original, residual_col, baseline_col, actual_col in zip(
            transform_result.original_target_columns,
            transform_result.residual_columns,
            transform_result.baseline_columns,
            transform_result.actual_columns,
        ):
            result[f"pred_residual_{original}"] = result[f"pred_{residual_col}"]
            result[f"actual_residual_{original}"] = result[residual_col]
            result[f"baseline_{original}"] = result[baseline_col]
            result[f"pred_{original}"] = result[f"pred_{residual_col}"] + result[baseline_col]
            result[f"actual_{original}"] = result[actual_col]
    else:
        for original, actual_col in zip(
            transform_result.original_target_columns,
            transform_result.actual_columns,
        ):
            if actual_col != original:
                result[f"pred_{original}"] = result[f"pred_{actual_col}"]
                result[f"actual_{original}"] = result[actual_col]

    result["target_mode"] = transform_result.target_mode
    result["target_horizon_rows"] = transform_result.target_horizon_rows
    if transform_result.residual_baseline_type:
        result["residual_baseline_type"] = transform_result.residual_baseline_type
        result["residual_baseline_window_rows"] = transform_result.residual_baseline_window_rows
    return result


def transformed_helper_columns(transform_result: TargetTransformResult) -> set[str]:
    return set(
        transform_result.actual_columns
        + transform_result.target_columns
        + transform_result.residual_columns
        + transform_result.baseline_columns
        + transform_result.original_target_columns
    )


def _forward_rolling_sum(series: pd.Series, horizon_rows: int) -> pd.Series:
    parts = [series.shift(-offset) for offset in range(horizon_rows)]
    return pd.concat(parts, axis=1).sum(axis=1, min_count=horizon_rows)
