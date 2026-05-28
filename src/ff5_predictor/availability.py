from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ff5_predictor.io import normalize_datetime_index


@dataclass(frozen=True)
class ReleaseGap:
    cutoff_date: pd.Timestamp
    target_dates: pd.DatetimeIndex
    gap_size: int


def latest_official_factor_date(ff5_df: pd.DataFrame) -> pd.Timestamp:
    ff5 = normalize_datetime_index(ff5_df)
    if ff5.empty:
        raise ValueError("FF5 dataframe is empty")
    return pd.Timestamp(ff5.index.max())


def latest_market_date(market_df: pd.DataFrame) -> pd.Timestamp:
    market = normalize_datetime_index(market_df)
    if market.empty:
        raise ValueError("Market dataframe is empty")
    return pd.Timestamp(market.index.max())


def unreleased_market_dates(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DatetimeIndex:
    official_date = latest_official_factor_date(ff5_df)
    market = normalize_datetime_index(market_df)
    return pd.DatetimeIndex(market.index[market.index > official_date])


def make_release_gap_splits(
    dates: pd.DatetimeIndex,
    gap_sizes: list[int],
    min_train_rows: int,
    step_rows: int,
) -> list[ReleaseGap]:
    clean_dates = pd.DatetimeIndex(pd.to_datetime(dates).tz_localize(None)).sort_values().unique()
    if len(clean_dates) == 0:
        return []
    gap_sizes = sorted({int(size) for size in gap_sizes if int(size) > 0})
    step = max(int(step_rows), 1)
    splits: list[ReleaseGap] = []
    max_gap = max(gap_sizes) if gap_sizes else 0
    for cutoff_pos in range(max(int(min_train_rows) - 1, 0), len(clean_dates) - max_gap, step):
        cutoff_date = pd.Timestamp(clean_dates[cutoff_pos])
        for gap_size in gap_sizes:
            start = cutoff_pos + 1
            end = start + gap_size
            if end <= len(clean_dates):
                splits.append(
                    ReleaseGap(
                        cutoff_date=cutoff_date,
                        target_dates=pd.DatetimeIndex(clean_dates[start:end]),
                        gap_size=gap_size,
                    )
                )
    return splits
