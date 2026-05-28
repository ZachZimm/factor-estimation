from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ff5_predictor.io import ensure_dir


def experiment_name(config: dict[str, Any]) -> str:
    experiments = config.get("experiments", {})
    experiment = config.get("experiment", {})
    return experiments.get("run_name") or experiment.get("name") or "baseline_research"


def experiment_root(config: dict[str, Any]) -> Path:
    output_dir = config.get("experiments", {}).get(
        "output_dir",
        config.get("experiment", {}).get("output_dir", "data/experiments"),
    )
    return Path(output_dir) / experiment_name(config)


def write_experiment_config(config: dict[str, Any]) -> None:
    root = ensure_dir(experiment_root(config))
    with (root / "config_resolved.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=True)


def write_predictions(config: dict[str, Any], model_type: str, predictions: pd.DataFrame) -> Path:
    path = ensure_dir(experiment_root(config) / "predictions") / f"{model_type}.csv"
    predictions.to_csv(path, index=False)
    return path


def write_metrics(config: dict[str, Any], name: str, metrics: dict[str, Any]) -> Path:
    path = ensure_dir(experiment_root(config) / "metrics") / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True, default=str)
    return path
