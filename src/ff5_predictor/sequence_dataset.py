from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from torch.utils.data import Dataset as TorchDataset
except Exception:  # pragma: no cover - exercised only when torch is unavailable
    TorchDataset = object


@dataclass(frozen=True)
class SequenceArrays:
    X: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    feature_columns: list[str]
    target_columns: list[str]
    target_positions: np.ndarray


def build_sequence_arrays(
    modeling_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    lookback_rows: int,
    min_sequence_rows: int | None = None,
) -> SequenceArrays:
    lookback = int(lookback_rows)
    min_rows = lookback if min_sequence_rows is None else int(min_sequence_rows)
    if min_rows != lookback:
        # The first implementation intentionally skips incomplete windows to keep
        # sequence leakage checks straightforward.
        min_rows = lookback

    df = modeling_df.sort_index()
    X_values = df[feature_columns].to_numpy(dtype=np.float32)
    y_values = df[target_columns].to_numpy(dtype=np.float32)
    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    positions: list[int] = []
    for i in range(min_rows, len(df)):
        start = i - lookback
        end = i
        X.append(X_values[start:end])
        y.append(y_values[i])
        dates.append(df.index[i])
        positions.append(i)

    if X:
        X_array = np.stack(X).astype(np.float32)
        y_array = np.stack(y).astype(np.float32)
    else:
        X_array = np.empty((0, lookback, len(feature_columns)), dtype=np.float32)
        y_array = np.empty((0, len(target_columns)), dtype=np.float32)
    return SequenceArrays(
        X=X_array,
        y=y_array,
        dates=pd.DatetimeIndex(dates),
        feature_columns=feature_columns,
        target_columns=target_columns,
        target_positions=np.asarray(positions, dtype=int),
    )


class FF5SequenceDataset(TorchDataset):
    def __init__(self, arrays: SequenceArrays, positions: np.ndarray):
        import torch

        self.arrays = arrays
        self.positions = np.asarray(positions, dtype=int)
        self._torch = torch

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int):
        pos = int(self.positions[idx])
        return (
            self._torch.from_numpy(self.arrays.X[pos]),
            self._torch.from_numpy(self.arrays.y[pos]),
        )
