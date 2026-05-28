from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from ff5_predictor.rolling_train import prediction_output_columns


def _baseline_records(
    modeling_df: pd.DataFrame,
    target_columns: list[str],
    min_train_rows: int,
    model_type: str,
    predict_fn: Callable[[pd.DataFrame], pd.Series],
    window_rows: int | None = None,
    step_size: int = 1,
) -> pd.DataFrame:
    df = modeling_df.sort_index()
    records: list[dict[str, Any]] = []
    eligible_counter = 0
    for i in range(len(df)):
        start = 0 if window_rows is None else max(0, i - window_rows)
        train_df = df.iloc[start:i]
        if len(train_df) < min_train_rows:
            continue
        if eligible_counter % step_size != 0:
            eligible_counter += 1
            continue
        eligible_counter += 1
        target_date = df.index[i]
        if not train_df.index.max() < target_date:
            raise AssertionError("Baseline training window leaked target date")
        pred = predict_fn(train_df[target_columns])
        record: dict[str, Any] = {
            "date": target_date,
            "model_type": model_type,
            "train_start_date": train_df.index[0],
            "train_end_date": train_df.index[-1],
            "n_train_rows": len(train_df),
        }
        for column in target_columns:
            record[f"pred_{column}"] = float(pred[column])
            record[f"actual_{column}"] = float(df.iloc[i][column])
        records.append(record)
    return pd.DataFrame.from_records(records, columns=prediction_output_columns(target_columns))


def rolling_mean_baseline(
    modeling_df: pd.DataFrame,
    target_columns: list[str],
    window_rows: int,
    min_train_rows: int,
    step_size: int = 1,
) -> pd.DataFrame:
    return _baseline_records(
        modeling_df,
        target_columns,
        min_train_rows,
        f"rolling_mean_{window_rows}",
        lambda y: y.tail(window_rows).mean(),
        window_rows=window_rows,
        step_size=step_size,
    )


def rolling_median_baseline(
    modeling_df: pd.DataFrame,
    target_columns: list[str],
    window_rows: int,
    min_train_rows: int,
    step_size: int = 1,
) -> pd.DataFrame:
    return _baseline_records(
        modeling_df,
        target_columns,
        min_train_rows,
        f"rolling_median_{window_rows}",
        lambda y: y.tail(window_rows).median(),
        window_rows=window_rows,
        step_size=step_size,
    )


def ewma_baseline(
    modeling_df: pd.DataFrame,
    target_columns: list[str],
    span: int,
    min_train_rows: int,
    step_size: int = 1,
) -> pd.DataFrame:
    return _baseline_records(
        modeling_df,
        target_columns,
        min_train_rows,
        f"ewma_{span}",
        lambda y: y.ewm(span=span, adjust=False).mean().iloc[-1],
        window_rows=None,
        step_size=step_size,
    )
