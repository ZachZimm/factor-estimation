from __future__ import annotations

import pandas as pd

from ff5_predictor.availability import (
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
