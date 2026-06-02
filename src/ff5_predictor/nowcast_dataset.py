from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ff5_predictor.availability import (
    filter_dates_by_config,
    latest_market_date,
    latest_official_factor_date,
    unreleased_market_dates,
)
from ff5_predictor.io import normalize_datetime_index
from ff5_predictor.nowcast_features import build_nowcast_features


@dataclass(frozen=True)
class NowcastDataset:
    train_df: pd.DataFrame
    inference_df: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]
    latest_official_date: pd.Timestamp
    latest_market_date: pd.Timestamp
    unreleased_dates: pd.DatetimeIndex
    metadata: dict[str, Any]


def build_nowcast_dataset(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: dict[str, Any],
) -> NowcastDataset:
    ff5 = normalize_datetime_index(ff5_df)
    market = normalize_datetime_index(market_df)
    target_columns = list(config["prediction"]["target_columns"])
    official_date = latest_official_factor_date(ff5)
    market_date = latest_market_date(market)
    all_unreleased = unreleased_market_dates(ff5, market)
    unreleased = filter_dates_by_config(all_unreleased, config)

    train_features = build_nowcast_features(ff5, market.loc[:official_date], config)
    train_df = train_features.features.join(ff5[target_columns], how="inner")
    train_df = train_df.loc[:official_date]
    feature_columns = list(train_features.feature_columns)
    train_df = train_df.dropna(subset=feature_columns + target_columns)

    inference_df = pd.DataFrame(index=unreleased)
    if len(unreleased) > 0:
        inference_features = build_nowcast_features(
            ff5,
            market.loc[:market_date],
            config,
            official_cutoff_date=official_date,
        )
        inference_df = inference_features.features.reindex(unreleased)
        inference_df = inference_df.dropna(subset=[col for col in feature_columns if col in inference_df.columns])
        missing_cols = [col for col in feature_columns if col not in inference_df.columns]
        if missing_cols:
            for col in missing_cols:
                inference_df[col] = pd.NA
            inference_df = inference_df.dropna(subset=feature_columns)
        inference_df = inference_df[feature_columns]

    return NowcastDataset(
        train_df=normalize_datetime_index(train_df),
        inference_df=normalize_datetime_index(inference_df),
        feature_columns=feature_columns,
        target_columns=target_columns,
        latest_official_date=official_date,
        latest_market_date=market_date,
        unreleased_dates=unreleased,
        metadata={
            "latest_official_factor_date": str(official_date.date()),
            "latest_market_date": str(market_date.date()),
            "n_all_unreleased_dates": int(len(all_unreleased)),
            "n_unreleased_dates": int(len(unreleased)),
            "n_train_rows": int(len(train_df)),
            "n_inference_rows": int(len(inference_df)),
            "date_filter": dict(config.get("date_filter", {})),
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "feature_metadata": train_features.metadata,
        },
    )
