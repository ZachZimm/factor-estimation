from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")

from ff5_predictor.torch_train import train_torch_walk_forward


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def test_torch_walk_forward_tiny_cpu_training_outputs_predictions() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = {"feature": [float(i) for i in range(30)]}
    for target in TARGETS:
        data[target] = [float(i) / 100 for i in range(30)]
    df = pd.DataFrame(data, index=dates)
    cfg = {
        "experiments": {"random_seed": 1, "output_dir": "data/experiments", "run_name": "test"},
        "walk_forward": {
            "checkpoint_frequency": "D",
            "train_window_rows": 9,
            "min_train_rows": 5,
            "min_fit_rows": 5,
            "validation_window_rows": 2,
            "require_validation": True,
        },
        "sequence": {"lookback_rows": 3, "batch_size": 4, "num_workers": 0},
        "torch": {
            "device": "cpu",
            "max_epochs": 2,
            "patience": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "standardize_targets": True,
            "deterministic": True,
            "restore_best_checkpoint": True,
        },
        "models": {"mlp_window": {"hidden_sizes": [8], "activation": "gelu", "dropout": 0.0}},
    }

    result = train_torch_walk_forward(df, ["feature"], TARGETS, "mlp_window", cfg)

    assert not result.predictions.empty
    assert (pd.to_datetime(result.predictions["train_end_date"]) < pd.to_datetime(result.predictions["date"])).all()
    assert {"checkpoint_id", "epoch_count", "best_epoch", "device", "lookback_rows"}.issubset(result.predictions.columns)
    assert result.predictions["best_validation_rmse"].notna().all()
    assert (result.predictions["n_validation_rows"] > 0).all()
