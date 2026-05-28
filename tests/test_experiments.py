from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")

import ff5_predictor.experiment_runner as experiment_runner
from ff5_predictor.experiment_runner import run_experiment


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def test_run_experiment_writes_outputs(tmp_path, monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = {"feature": [float(i) for i in range(30)]}
    for target in TARGETS:
        data[target] = [float(i) / 100 for i in range(30)]
    df = pd.DataFrame(data, index=dates)
    monkeypatch.setattr(experiment_runner, "_load_modeling_dataset", lambda config: df)
    cfg = {
        "prediction": {"target_columns": TARGETS},
        "experiments": {
            "output_dir": str(tmp_path),
            "run_name": "tiny",
            "random_seed": 1,
            "models": ["rolling_mean", "ewma", "mlp_window"],
        },
        "walk_forward": {
            "checkpoint_frequency": "D",
            "train_window_rows": 8,
            "min_train_rows": 5,
            "validation_window_rows": 1,
        },
        "training": {"step_size": 1, "scale_features": True},
        "sequence": {"lookback_rows": 3, "batch_size": 4, "num_workers": 0},
        "torch": {
            "device": "cpu",
            "max_epochs": 1,
            "patience": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "standardize_targets": True,
            "deterministic": True,
        },
        "models": {
            "ewma": {"default_span": 3},
            "mlp_window": {"hidden_sizes": [8], "activation": "gelu", "dropout": 0.0},
        },
    }

    result = run_experiment(cfg)

    assert "metrics" in result
    assert (tmp_path / "tiny" / "predictions" / "all_predictions.csv").exists()
    assert (tmp_path / "tiny" / "metrics" / "metrics.json").exists()
