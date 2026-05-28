from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    checkpoint_date: pd.Timestamp
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp | None
    validation_end_date: pd.Timestamp | None
    predict_start_date: pd.Timestamp
    predict_end_date: pd.Timestamp
    train_positions: np.ndarray
    validation_positions: np.ndarray
    prediction_positions: np.ndarray
    n_fit_rows: int
    n_validation_rows: int


def make_walk_forward_splits(index: pd.DatetimeIndex, config: dict[str, Any]) -> list[WalkForwardSplit]:
    wf = config.get("walk_forward", {})
    return build_walk_forward_checkpoints(
        dates=index,
        train_window_rows=int(wf.get("train_window_rows", wf.get("train_window_days", 1260))),
        min_train_rows=int(wf.get("min_fit_rows", wf.get("min_train_rows", 1000))),
        validation_window_rows=int(wf.get("validation_window_rows", 252)),
        retrain_frequency=str(wf.get("retrain_frequency", wf.get("checkpoint_frequency", "monthly"))),
        require_validation=bool(wf.get("require_validation", False)),
    )


def build_walk_forward_checkpoints(
    dates: pd.DatetimeIndex,
    train_window_rows: int,
    min_train_rows: int,
    validation_window_rows: int,
    retrain_frequency: str,
    require_validation: bool = False,
) -> list[WalkForwardSplit]:
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    if dates.empty:
        return []
    checkpoint_positions = _checkpoint_positions(dates, retrain_frequency)
    splits: list[WalkForwardSplit] = []
    for pos_idx, checkpoint_pos in enumerate(checkpoint_positions):
        checkpoint_date = dates[checkpoint_pos]
        history = np.arange(max(0, checkpoint_pos - train_window_rows), checkpoint_pos)
        required_history = min_train_rows + validation_window_rows if require_validation else min_train_rows
        if len(history) < required_history:
            continue

        next_checkpoint = (
            checkpoint_positions[pos_idx + 1] if pos_idx + 1 < len(checkpoint_positions) else len(dates)
        )
        prediction_positions = np.arange(checkpoint_pos, next_checkpoint)
        if len(prediction_positions) == 0:
            continue

        if validation_window_rows > 0 and len(history) - validation_window_rows >= min_train_rows:
            validation_positions = history[-validation_window_rows:]
            train_positions = history[:-validation_window_rows]
        else:
            if require_validation:
                continue
            validation_positions = np.array([], dtype=int)
            train_positions = history
        if len(train_positions) < min_train_rows:
            continue
        if require_validation and len(validation_positions) != validation_window_rows:
            continue

        if not dates[train_positions].max() < dates[prediction_positions].min():
            raise AssertionError("Walk-forward train/prediction overlap detected")
        if len(validation_positions) and not dates[validation_positions].max() < dates[prediction_positions].min():
            raise AssertionError("Walk-forward validation/prediction overlap detected")

        splits.append(
            WalkForwardSplit(
                checkpoint_date=checkpoint_date,
                train_start_date=dates[train_positions[0]],
                train_end_date=dates[train_positions[-1]],
                validation_start_date=dates[validation_positions[0]] if len(validation_positions) else None,
                validation_end_date=dates[validation_positions[-1]] if len(validation_positions) else None,
                predict_start_date=dates[prediction_positions[0]],
                predict_end_date=dates[prediction_positions[-1]],
                train_positions=train_positions,
                validation_positions=validation_positions,
                prediction_positions=prediction_positions,
                n_fit_rows=int(len(train_positions)),
                n_validation_rows=int(len(validation_positions)),
            )
        )
    return splits


def _checkpoint_positions(dates: pd.DatetimeIndex, frequency: str) -> list[int]:
    normalized = frequency.lower()
    if normalized in {"d", "day", "daily"}:
        return list(range(len(dates)))
    if normalized in {"w", "week", "weekly"}:
        periods = dates.to_period("W")
    elif normalized in {"m", "month", "monthly"}:
        periods = dates.to_period("M")
    elif normalized in {"q", "quarter", "quarterly"}:
        periods = dates.to_period("Q")
    else:
        periods = dates.to_period(frequency)
    first_positions = pd.Series(np.arange(len(dates)), index=periods).groupby(level=0).min()
    return [int(pos) for pos in first_positions.to_list()]
