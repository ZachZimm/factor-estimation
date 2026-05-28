from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import pandas as pd

from ff5_predictor.baselines import ewma_baseline, rolling_mean_baseline, rolling_median_baseline
from ff5_predictor.data_famafrench import load_ff5
from ff5_predictor.data_yfinance import load_market_data
from ff5_predictor.dataset import build_modeling_dataset, split_feature_target_columns
from ff5_predictor.evaluation import evaluate_on_shared_dates, evaluate_prediction_groups
from ff5_predictor.evaluation import rank_models
from ff5_predictor.experiment_config import TABULAR_MODEL_NAMES, TORCH_MODEL_NAMES
from ff5_predictor.experiment_io import experiment_root
from ff5_predictor.experiment_io import write_experiment_config, write_metrics, write_predictions
from ff5_predictor.regime_evaluation import add_regime_labels, evaluate_by_regime
from ff5_predictor.rolling_train import naive_previous_value_baseline, rolling_predict
from ff5_predictor.target_transform import (
    apply_target_transform,
    reconstruct_predictions,
    transformed_helper_columns,
)

LOGGER = logging.getLogger(__name__)


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    models = list(config.get("experiments", {}).get("models", config.get("experiment", {}).get("run_models", [])))
    if not models:
        raise ValueError("No experiment models configured")
    modeling_df = _load_modeling_dataset(config)
    base_target_columns = list(config["prediction"]["target_columns"])
    transform = apply_target_transform(modeling_df, base_target_columns, config)
    train_df = transform.modeling_df
    excluded = transformed_helper_columns(transform)
    feature_columns = [col for col in train_df.columns if col not in excluded]
    target_columns = transform.target_columns
    all_predictions: list[pd.DataFrame] = []
    checkpoint_metrics: dict[str, Any] = {}
    write_experiment_config(config)

    for model_type in models:
        LOGGER.info("Running experiment model: %s", model_type)
        predictions, extra = train_one_model(
            model_type=model_type,
            modeling_df=train_df,
            feature_columns=feature_columns,
            target_columns=target_columns,
            config=config,
        )
        predictions = reconstruct_predictions(predictions, transform, config)
        write_predictions(config, _output_model_name(model_type, predictions), predictions)
        all_predictions.append(predictions)
        if extra:
            checkpoint_metrics[model_type] = extra

    combined = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    write_predictions(config, "all_predictions", combined)
    metrics = evaluate_prediction_groups(combined, base_target_columns)
    shared_metrics = evaluate_on_shared_dates(combined, base_target_columns, baseline_model="rolling_mean")
    ranking = rank_models(shared_metrics)
    ranking_path = experiment_root(config) / "metrics" / "model_ranking.csv"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(ranking_path, index=False)
    regime_predictions = add_regime_labels(combined, train_df, config)
    regime_metrics = evaluate_by_regime(regime_predictions, base_target_columns, config)
    if checkpoint_metrics:
        metrics["_checkpoint_metrics"] = checkpoint_metrics
    write_metrics(config, "metrics.json", metrics)
    write_metrics(config, "shared_date_metrics.json", shared_metrics)
    write_metrics(config, "regime_metrics.json", regime_metrics)
    return {
        "metrics": metrics,
        "shared_date_metrics": shared_metrics,
        "regime_metrics": regime_metrics,
    }


def train_one_model(
    model_type: str,
    modeling_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Any]:
    wf = config.get("walk_forward", {})
    train_window_rows = int(wf.get("train_window_rows", config.get("training", {}).get("train_window_days", 1260)))
    min_train_rows = int(wf.get("min_train_rows", config.get("training", {}).get("min_train_rows", 1000)))
    step_size = int(config.get("training", {}).get("step_size", 1))

    if model_type == "naive_previous":
        return naive_previous_value_baseline(modeling_df, target_columns, step_size=step_size), None
    if model_type == "rolling_mean":
        return rolling_mean_baseline(modeling_df, target_columns, train_window_rows, min_train_rows, step_size), None
    if model_type == "rolling_median":
        return rolling_median_baseline(modeling_df, target_columns, train_window_rows, min_train_rows, step_size), None
    if model_type == "ewma":
        span = int(config.get("models", {}).get("ewma", {}).get("default_span", 21))
        return ewma_baseline(modeling_df, target_columns, span, min_train_rows, step_size), None
    if model_type in TABULAR_MODEL_NAMES:
        model_config = deepcopy(config)
        model_config.setdefault("training", {})
        model_config["training"]["model_type"] = model_type
        model_config["training"]["train_window_days"] = train_window_rows
        model_config["training"]["min_train_rows"] = min_train_rows
        return rolling_predict(modeling_df, feature_columns, target_columns, model_config), None
    if model_type in TORCH_MODEL_NAMES:
        from ff5_predictor.torch_train import train_torch_walk_forward

        result = train_torch_walk_forward(modeling_df, feature_columns, target_columns, model_type, config)
        return result.predictions, result.checkpoint_metrics
    raise ValueError(f"Unsupported experiment model: {model_type}")


def _load_modeling_dataset(config: dict[str, Any]) -> pd.DataFrame:
    ff5_df = load_ff5(config)
    market_df = load_market_data(config)
    modeling_df = build_modeling_dataset(ff5_df, market_df, config)
    if modeling_df.empty:
        raise ValueError("Experiment modeling dataset is empty")
    return modeling_df


def _output_model_name(model_type: str, predictions: pd.DataFrame) -> str:
    if not predictions.empty and "model_type" in predictions:
        unique = predictions["model_type"].dropna().unique()
        if len(unique) == 1:
            return str(unique[0])
    return model_type
