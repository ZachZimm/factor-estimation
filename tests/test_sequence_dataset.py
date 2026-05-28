from __future__ import annotations

import pandas as pd

from ff5_predictor.sequence_dataset import build_sequence_arrays


def test_sequence_sample_uses_only_rows_before_target() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame({"feature": range(6), "target": range(10, 16)}, index=dates)

    arrays = build_sequence_arrays(df, ["feature"], ["target"], lookback_rows=3)

    assert arrays.X.shape == (3, 3, 1)
    assert arrays.y.shape == (3, 1)
    assert arrays.dates[0] == pd.Timestamp("2024-01-04")
    assert arrays.X[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert arrays.y[0, 0] == 13.0
    assert arrays.target_positions.tolist() == [3, 4, 5]
