from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ff5_predictor.attribution import explain_ridge_predictions
from ff5_predictor.data_famafrench import load_ff5
from ff5_predictor.data_yfinance import load_market_data
from ff5_predictor.io import ensure_dir
from ff5_predictor.nowcast_dataset import build_nowcast_dataset
from ff5_predictor.nowcast_engine import (
    NowcastTargetSpec,
    empty_production_predictions,
    run_nowcast_engine,
    select_production_columns,
)
from ff5_predictor.nowcast_io import (
    create_nowcast_run_dir,
    sync_latest_copy,
    write_json,
    write_nowcast_predictions,
    write_yaml,
)
from ff5_predictor.nowcast_models import (
    FittedNowcastModel,
    save_fitted_model,
)


@dataclass(frozen=True)
class ProductionNowcastResult:
    predictions: pd.DataFrame
    feature_snapshot: pd.DataFrame
    metadata: dict[str, Any]
    run_dir: Path


def run_production_nowcast(config: dict[str, Any]) -> ProductionNowcastResult:
    ff5_df = load_ff5(config)
    market_df = load_market_data(config)
    return run_production_nowcast_from_frames(ff5_df, market_df, config)


def run_production_nowcast_from_frames(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: dict[str, Any],
) -> ProductionNowcastResult:
    run_dir = create_nowcast_run_dir(config)
    write_yaml(run_dir / "config_resolved.yaml", config)
    dataset = build_nowcast_dataset(ff5_df, market_df, config)
    write_json(run_dir / "metadata" / "dataset_metadata.json", dataset.metadata)

    spec = NowcastTargetSpec(
        target_dates=dataset.unreleased_dates,
        cutoff_date=dataset.latest_official_date,
        latest_market_date=dataset.latest_market_date,
        actuals=None,
        is_unreleased=True,
        release_gap_size_by_date=None,
    )
    engine_result = run_nowcast_engine(
        ff5_df=ff5_df,
        market_df=market_df,
        train_df=dataset.train_df,
        feature_columns=dataset.feature_columns,
        target_columns=dataset.target_columns,
        spec=spec,
        config=config,
    )
    predictions = select_production_columns(engine_result.predictions, dataset.target_columns)
    if len(dataset.unreleased_dates) == 0:
        predictions = empty_production_predictions(dataset.target_columns)
    feature_snapshot = engine_result.feature_snapshots

    ridge_model = engine_result.fitted_models.get("ridge")
    if isinstance(ridge_model, FittedNowcastModel) and bool(config.get("nowcast", {}).get("save_model_artifact", True)):
        save_primary_model_artifact(ridge_model, run_dir)

    write_nowcast_predictions(run_dir, "latest_nowcast.csv", predictions)
    write_json(run_dir / "predictions" / "latest_nowcast.json", {"records": predictions.to_dict(orient="records")})
    if bool(config.get("nowcast", {}).get("save_feature_snapshot", True)):
        ensure_dir(run_dir / "features")
        feature_snapshot.to_parquet(run_dir / "features" / "latest_feature_snapshot.parquet")

    attribution_metadata: dict[str, Any] = {"enabled": False}
    if isinstance(ridge_model, FittedNowcastModel) and bool(config.get("nowcast", {}).get("save_feature_attributions", False)):
        attribution_metadata = save_ridge_attributions(ridge_model, feature_snapshot, predictions, config, run_dir)

    metadata = {
        **dataset.metadata,
        "n_prediction_rows": int(len(predictions)),
        "models": config.get("nowcast", {}).get("models", []),
        "primary_model": config.get("nowcast", {}).get("primary_model"),
        "run_dir": str(run_dir),
        "production_profile": "market_only_all_candidates",
        "feature_policy": {
            "uses_ff5_input_features": bool(config.get("target_features", {}).get("include_lagged_targets", False)),
            "uses_recursive_factor_lags": bool(config.get("availability", {}).get("recursive_factor_lags", True)),
            "uses_same_day_market_data": int(config.get("availability", {}).get("market_data_lag_rows", 0)) == 0,
        },
        "attribution": attribution_metadata,
    }
    write_json(run_dir / "metadata" / "metadata.json", metadata)
    if config.get("nowcast", {}).get("run_name") == "production_latest":
        sync_latest_copy(config, run_dir)
    return ProductionNowcastResult(
        predictions=predictions,
        feature_snapshot=feature_snapshot,
        metadata=metadata,
        run_dir=run_dir,
    )


def save_ridge_attributions(
    fitted: FittedNowcastModel,
    feature_snapshot: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    top_n = int(config.get("attribution", {}).get("top_n", 20))
    result = explain_ridge_predictions(fitted, feature_snapshot, predictions, top_n=top_n)
    attribution_dir = ensure_dir(run_dir / "attribution")
    paths = {
        "coefficients": attribution_dir / "ridge_coefficients.csv",
        "contributions_long": attribution_dir / "ridge_contributions_long.parquet",
        "contributions_wide": attribution_dir / "ridge_contributions_wide.parquet",
        "top_contributions": attribution_dir / "ridge_top_contributions.csv",
        "group_contributions": attribution_dir / "ridge_group_contributions.csv",
        "metadata": attribution_dir / "attribution_metadata.json",
    }
    result.coefficient_table.to_csv(paths["coefficients"], index=False)
    result.contribution_long.to_parquet(paths["contributions_long"], index=False)
    result.contribution_wide.to_parquet(paths["contributions_wide"], index=False)
    result.top_contributions.to_csv(paths["top_contributions"], index=False)
    result.group_summary.to_csv(paths["group_contributions"], index=False)
    write_json(paths["metadata"], result.metadata)
    return {
        "enabled": True,
        "top_n": top_n,
        "paths": {key: str(path) for key, path in paths.items()},
        **result.metadata,
    }


def save_primary_model_artifact(
    fitted: FittedNowcastModel,
    run_dir: Path,
) -> None:
    ensure_dir(run_dir / "models")
    save_fitted_model(fitted, run_dir / "models" / f"{fitted.model_type}.joblib")
