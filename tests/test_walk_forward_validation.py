from __future__ import annotations

import pandas as pd

from ff5_predictor.walk_forward import build_walk_forward_checkpoints


def test_require_validation_skips_until_fit_and_validation_available() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    splits = build_walk_forward_checkpoints(
        dates=dates,
        train_window_rows=15,
        min_train_rows=10,
        validation_window_rows=5,
        retrain_frequency="D",
        require_validation=True,
    )
    assert splits
    first = splits[0]
    assert len(first.train_positions) == 10
    assert len(first.validation_positions) == 5
    assert dates[first.train_positions].max() < dates[first.validation_positions].min()
    assert dates[first.validation_positions].max() < dates[first.prediction_positions].min()
