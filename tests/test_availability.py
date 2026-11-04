from __future__ import annotations

import pandas as pd

from ff5_predictor.availability import (
    filter_dates_by_config,
    latest_market_date,
    latest_official_factor_date,
    make_release_gap_splits,
    unreleased_market_dates,
)


def test_availability_dates_and_release_gaps() -> None:
    ff5 = pd.DataFrame({"Mkt-RF": [1, 2]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    market = pd.DataFrame({"SPY_close": [1, 2, 3]}, index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))

    assert latest_official_factor_date(ff5) == pd.Timestamp("2024-01-03")
    assert latest_market_date(market) == pd.Timestamp("2024-01-04")
    assert list(unreleased_market_dates(ff5, market)) == [pd.Timestamp("2024-01-04")]

    splits = make_release_gap_splits(pd.date_range("2024-01-01", periods=8), [1, 3], min_train_rows=3, step_rows=2)
    assert splits[0].cutoff_date == pd.Timestamp("2024-01-03")
    assert all(date > split.cutoff_date for split in splits for date in split.target_dates)
    assert {split.gap_size for split in splits} == {1, 3}


def test_filter_dates_by_config() -> None:
    dates = pd.date_range("2024-01-01", periods=5)
    filtered = filter_dates_by_config(
        dates,
        {"date_filter": {"start_date": "2024-01-02", "end_date": "2024-01-04"}},
    )

    assert list(filtered) == list(pd.date_range("2024-01-02", "2024-01-04"))


def test_longer_horizons_do_not_remove_short_gap_cutoffs() -> None:
    dates = pd.bdate_range("2024-01-01", periods=100)
    short = make_release_gap_splits(dates, [1, 10], min_train_rows=20, step_rows=7)
    expanded = make_release_gap_splits(dates, [1, 10, 21, 42, 63], min_train_rows=20, step_rows=7)
    short_keys = {(split.cutoff_date, split.gap_size) for split in short}
    expanded_keys = {(split.cutoff_date, split.gap_size) for split in expanded if split.gap_size <= 10}
    assert short_keys == expanded_keys
    assert max(split.gap_size for split in expanded) == 63
