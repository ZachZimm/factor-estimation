from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ff5_predictor.io import ensure_dir
from ff5_predictor.models import make_model, make_scaler


def prediction_output_columns(target_columns: list[str]) -> list[str]:
    return (
        ["date"]
        + [f"pred_{column}" for column in target_columns]
        + [f"actual_{column}" for column in target_columns]
        + ["model_type", "train_start_date", "train_end_date", "n_train_rows"]
    )


def rolling_predict(
    modeling_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    df = modeling_df.sort_index()
    train_window = int(config["training"]["train_window_days"])
    min_train_rows = int(config["training"]["min_train_rows"])
    step_size = int(config["training"].get("step_size", 1))
    model_type = config["training"].get("model_type", "ridge")
    scale_features = bool(config["training"].get("scale_features", True))
    save_models = bool(config["training"].get("save_models", False))
    models_dir = Path(config["output"].get("models_dir", "data/models"))

    records: list[dict[str, Any]] = []
    eligible_counter = 0
    for i in range(len(df)):
        train_start_pos = max(0, i - train_window)
        train_df = df.iloc[train_start_pos:i]
        if len(train_df) < min_train_rows:
            continue
        if eligible_counter % step_size != 0:
            eligible_counter += 1
            continue
        eligible_counter += 1

        target_date = df.index[i]
        if not (train_df.index.max() < target_date):
            raise AssertionError("Training window leaked target date or future rows")

        X_train = train_df[feature_columns]
        y_train = train_df[target_columns]
        X_target = df.iloc[[i]][feature_columns]

        scaler = make_scaler(scale_features)
        if scaler is not None:
            X_train_model = scaler.fit_transform(X_train)
            X_target_model = scaler.transform(X_target)
        else:
            X_train_model = X_train.to_numpy()
            X_target_model = X_target.to_numpy()

        model = make_model(model_type, config)
        model.fit(X_train_model, y_train)
        prediction = model.predict(X_target_model)[0]

        record: dict[str, Any] = {
            "date": target_date,
            "model_type": model_type,
            "train_start_date": train_df.index[0],
            "train_end_date": train_df.index[-1],
            "n_train_rows": len(train_df),
        }
        for column, value in zip(target_columns, prediction):
            record[f"pred_{column}"] = float(value)
        for column in target_columns:
            record[f"actual_{column}"] = float(df.iloc[i][column])
        records.append(record)

        if save_models:
            ensure_dir(models_dir)
            artifact_path = models_dir / f"{model_type}_{target_date:%Y%m%d}.joblib"
            joblib.dump(
                {
                    "scaler": scaler,
                    "model": model,
                    "feature_columns": feature_columns,
                    "target_columns": target_columns,
                    "train_start_date": train_df.index[0],
                    "train_end_date": train_df.index[-1],
                },
                artifact_path,
            )

    return pd.DataFrame.from_records(records, columns=prediction_output_columns(target_columns))


def naive_previous_value_baseline(
    modeling_df: pd.DataFrame,
    target_columns: list[str],
    step_size: int = 1,
) -> pd.DataFrame:
    df = modeling_df.sort_index()
    records: list[dict[str, Any]] = []
    eligible_counter = 0
    for i in range(1, len(df)):
        if eligible_counter % step_size != 0:
            eligible_counter += 1
            continue
        eligible_counter += 1
        target_date = df.index[i]
        prev_date = df.index[i - 1]
        record: dict[str, Any] = {
            "date": target_date,
            "model_type": "naive_previous",
            "train_start_date": prev_date,
            "train_end_date": prev_date,
            "n_train_rows": 1,
        }
        for column in target_columns:
            record[f"pred_{column}"] = float(df.iloc[i - 1][column])
            record[f"actual_{column}"] = float(df.iloc[i][column])
        records.append(record)
    return pd.DataFrame.from_records(records, columns=prediction_output_columns(target_columns))
