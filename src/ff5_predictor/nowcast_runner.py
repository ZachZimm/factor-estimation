from __future__ import annotations

from typing import Any

from ff5_predictor.latest_nowcast import run_latest_nowcast
from ff5_predictor.release_gap_backtest import run_release_gap_backtest


def run_nowcast(config: dict[str, Any]):
    return run_latest_nowcast(config)


def run_nowcast_backtest(config: dict[str, Any]):
    return run_release_gap_backtest(config)
