from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FundamentalsFeatureResult:
    features: pd.DataFrame
    metadata: dict[str, Any]


def build_fundamentals_features(
    config: dict[str, Any],
    as_of_dates: pd.DatetimeIndex,
) -> FundamentalsFeatureResult:
    enabled = bool(config.get("fundamentals", {}).get("enabled", False))
    metadata = {
        "enabled": enabled,
        "source_path": config.get("fundamentals", {}).get("source_path"),
        "point_in_time_policy": config.get("fundamentals", {}).get("point_in_time_policy"),
        "survivorship_bias_warning": "Fundamentals are deferred; future loaders must audit survivorship bias.",
        "coverage_by_date": {},
        "missingness_summary": {},
    }
    if enabled:
        LOGGER.warning("Fundamentals are configured as enabled, but no loader is implemented yet; returning no features")
    return FundamentalsFeatureResult(features=pd.DataFrame(index=as_of_dates), metadata=metadata)
