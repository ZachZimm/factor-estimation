from __future__ import annotations

import pandas as pd

from ff5_predictor.walk_forward import build_walk_forward_checkpoints


def test_walk_forward_dates_do_not_overlap() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    splits = build_walk_forward_checkpoints(
        dates=dates,
        train_window_rows=10,
        min_train_rows=6,
        validation_window_rows=2,
        retrain_frequency="weekly",
    )

    assert splits
    for split in splits:
        assert dates[split.train_positions].max() < dates[split.prediction_positions].min()
        if len(split.validation_positions):
            assert dates[split.train_positions].max() < dates[split.validation_positions].min()
            assert dates[split.validation_positions].max() < dates[split.prediction_positions].min()
