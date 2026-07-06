from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
import pandas as pd

from ff5_predictor.availability import filter_dates_by_config, make_release_gap_splits
from ff5_predictor.data_famafrench import load_ff5
from ff5_predictor.data_yfinance import load_market_data
from ff5_predictor.evaluation import evaluate_on_shared_dates, evaluate_prediction_groups, rank_models
from ff5_predictor.io import ensure_dir, normalize_datetime_index
from ff5_predictor.nowcast_engine import (
    NowcastTargetSpec,
    empty_backtest_predictions,
    run_nowcast_engine,
    select_backtest_columns,
)
from ff5_predictor.nowcast_features import build_nowcast_features
from ff5_predictor.nowcast_io import create_nowcast_run_dir, write_json, write_nowcast_predictions, write_yaml
from ff5_predictor.training_diagnostics import write_training_diagnostics


@dataclass(frozen=True)
class ReleaseGapBacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    gap_metrics: dict[str, Any]
    run_dir: Path
    training_history: pd.DataFrame


@dataclass(frozen=True)
class _CutoffBacktestResult:
    predictions: pd.DataFrame
    training_history: pd.DataFrame


def run_release_gap_backtest(config: dict[str, Any]) -> ReleaseGapBacktestResult:
    ff5_df = load_ff5(config)
    market_df = load_market_data(config)
    return run_release_gap_backtest_from_frames(ff5_df, market_df, config)


def run_release_gap_backtest_from_frames(
    ff5_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: dict[str, Any],
) -> ReleaseGapBacktestResult:
    ff5 = normalize_datetime_index(ff5_df)
    market = normalize_datetime_index(market_df)
    target_columns = list(config["prediction"]["target_columns"])
    aligned_dates = pd.DatetimeIndex(ff5.index.intersection(market.index)).sort_values()
    static_feature_frame = _build_static_feature_frame(ff5, market, config)
    min_rows = int(config.get("nowcast", {}).get("min_train_rows", config.get("training", {}).get("min_train_rows", 1000)))
    step_rows = int(config.get("nowcast", {}).get("backtest_step_rows", 21))
    gap_sizes = [int(v) for v in config.get("availability", {}).get("release_gap_backtest_days", [1, 2, 3, 5, 10])]
    splits = make_release_gap_splits(aligned_dates, gap_sizes, min_rows, step_rows)
    grouped: dict[pd.Timestamp, list] = {}
    for split in splits:
        grouped.setdefault(split.cutoff_date, []).append(split)

    tasks = []
    for cutoff_date, cutoff_splits in grouped.items():
        max_gap = max(split.gap_size for split in cutoff_splits)
        raw_target_dates = pd.DatetimeIndex(aligned_dates[(aligned_dates > cutoff_date)]).sort_values()[:max_gap]
        selected_target_dates = filter_dates_by_config(raw_target_dates, config)
        if len(selected_target_dates):
            if bool(config.get("availability", {}).get("recursive_factor_lags", True)):
                engine_target_dates = pd.DatetimeIndex(raw_target_dates[raw_target_dates <= selected_target_dates.max()])
            else:
                engine_target_dates = selected_target_dates
            tasks.append((cutoff_date, engine_target_dates, selected_target_dates, cutoff_splits))

    backtest_cfg = config.get("backtest", {})
    n_jobs = int(backtest_cfg.get("n_jobs", 1))
    if n_jobs == 1 or len(tasks) <= 1:
        results = [
            _predict_gap_for_cutoff(
                ff5,
                market,
                cutoff_date,
                target_dates,
                selected_dates,
                cutoff_splits,
                config,
                static_feature_frame,
            )
            for cutoff_date, target_dates, selected_dates, cutoff_splits in tasks
        ]
    else:
        results = Parallel(
            n_jobs=n_jobs,
            backend=str(backtest_cfg.get("backend", "loky")),
            verbose=int(backtest_cfg.get("verbose", 0)),
        )(
            delayed(_predict_gap_for_cutoff)(
                ff5,
                market,
                cutoff_date,
                target_dates,
                selected_dates,
                cutoff_splits,
                config,
                static_feature_frame,
            )
            for cutoff_date, target_dates, selected_dates, cutoff_splits in tasks
        )

    records: list[dict[str, Any]] = []
    training_histories: list[pd.DataFrame] = []
    for result in results:
        records.extend(result.predictions.to_dict(orient="records"))
        if not result.training_history.empty:
            training_histories.append(result.training_history)

    predictions = select_backtest_columns(pd.DataFrame(records), target_columns)
    training_history = pd.concat(training_histories, ignore_index=True) if training_histories else pd.DataFrame()
    run_dir = create_nowcast_run_dir(config)
    write_yaml(run_dir / "config_resolved.yaml", config)
    write_nowcast_predictions(run_dir, "release_gap_predictions.csv", predictions)
    residual_series = _model_implied_minus_official_series(predictions, target_columns)
    write_nowcast_predictions(run_dir, "model_implied_minus_official_series.csv", residual_series)
    metrics = evaluate_prediction_groups(predictions, target_columns) if not predictions.empty else {}
    shared = evaluate_on_shared_dates(predictions, target_columns, baseline_model="ewma") if not predictions.empty else {}
    ranking = rank_models(shared)
    ensure_dir(run_dir / "metrics")
    ranking.to_csv(run_dir / "metrics" / "model_ranking.csv", index=False)
    gap_metrics = _gap_metrics(predictions, target_columns)
    write_json(run_dir / "metrics" / "metrics.json", metrics)
    write_json(run_dir / "metrics" / "shared_date_metrics.json", shared)
    write_json(run_dir / "metrics" / "gap_metrics.json", gap_metrics)
    training_diagnostics_metadata = write_training_diagnostics(run_dir, training_history)
    write_json(run_dir / "metadata" / "backtest_metadata.json", _backtest_metadata(config, predictions, training_diagnostics_metadata))
    write_json(run_dir / "metadata" / "feature_extraction_metadata.json", _feature_extraction_metadata(config, predictions))
    return ReleaseGapBacktestResult(
        predictions=predictions,
        metrics=metrics,
        gap_metrics=gap_metrics,
        run_dir=run_dir,
        training_history=training_history,
    )


def _predict_gap_for_cutoff(
    ff5: pd.DataFrame,
    market: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    target_dates: pd.DatetimeIndex,
    selected_target_dates: pd.DatetimeIndex,
    cutoff_splits: list,
    config: dict[str, Any],
    static_feature_frame: pd.DataFrame | None = None,
) -> _CutoffBacktestResult:
    target_columns = list(config["prediction"]["target_columns"])
    train_df, feature_columns = _training_frame(ff5, market, cutoff_date, config, static_feature_frame)
    min_rows = int(config.get("nowcast", {}).get("min_train_rows", config.get("training", {}).get("min_train_rows", 1000)))
    if len(train_df) < min_rows:
        return _CutoffBacktestResult(empty_backtest_predictions(target_columns), pd.DataFrame())
    release_gap_size_by_date: dict[pd.Timestamp, list[int]] = {}
    for split in cutoff_splits:
        for date in split.target_dates:
            release_gap_size_by_date.setdefault(pd.Timestamp(date), []).append(int(split.gap_size))
    spec = NowcastTargetSpec(
        target_dates=target_dates,
        cutoff_date=pd.Timestamp(cutoff_date),
        latest_market_date=pd.Timestamp(target_dates.max()) if len(target_dates) else pd.Timestamp(cutoff_date),
        actuals=ff5.loc[target_dates, target_columns],
        is_unreleased=False,
        release_gap_size_by_date=release_gap_size_by_date,
    )
    result = run_nowcast_engine(
        ff5_df=ff5.loc[:cutoff_date],
        market_df=market.loc[: pd.Timestamp(target_dates.max())] if len(target_dates) else market.loc[:cutoff_date],
        train_df=train_df,
        feature_columns=feature_columns,
        target_columns=target_columns,
        spec=spec,
        config=config,
        feature_frame=static_feature_frame,
    )
    predictions = select_backtest_columns(result.predictions, target_columns)
    return _CutoffBacktestResult(
        predictions=_filter_prediction_frame(predictions, selected_target_dates),
        training_history=result.training_history,
    )


def _training_frame(
    ff5: pd.DataFrame,
    market: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    config: dict[str, Any],
    static_feature_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    target_columns = list(config["prediction"]["target_columns"])
    if static_feature_frame is not None:
        feature_frame = static_feature_frame.loc[:cutoff_date].dropna(axis=1, how="all")
        train_df = feature_frame.join(ff5[target_columns].loc[:cutoff_date], how="inner")
        feature_columns = list(feature_frame.columns)
    else:
        feature_result = build_nowcast_features(ff5.loc[:cutoff_date], market.loc[:cutoff_date], config)
        train_df = feature_result.features.join(ff5[target_columns].loc[:cutoff_date], how="inner")
        feature_columns = list(feature_result.feature_columns)
    train_df = train_df.dropna(subset=feature_columns + target_columns)
    return train_df, feature_columns


def _build_static_feature_frame(ff5: pd.DataFrame, market: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame | None:
    if not _can_use_static_feature_frame(config):
        return None
    feature_result = build_nowcast_features(ff5.iloc[0:0], market, config)
    return feature_result.features


def _can_use_static_feature_frame(config: dict[str, Any]) -> bool:
    return (
        not bool(config.get("target_features", {}).get("include_lagged_targets", False))
        and not bool(config.get("availability", {}).get("recursive_factor_lags", False))
        and not bool(config.get("fundamentals", {}).get("enabled", False))
    )


def _backtest_metadata(
    config: dict[str, Any],
    predictions: pd.DataFrame,
    training_diagnostics_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "nowcast_profile": "market_only_all_candidates",
        "n_tickers": int(len(config.get("data", {}).get("tickers", []))),
        "uses_ff5_input_features": bool(config.get("target_features", {}).get("include_lagged_targets", False)),
        "uses_recursive_factor_lags": bool(config.get("availability", {}).get("recursive_factor_lags", True)),
        "models": list(config.get("nowcast", {}).get("models", [])),
        "release_gap_backtest_days": list(config.get("availability", {}).get("release_gap_backtest_days", [])),
        "date_filter": dict(config.get("date_filter", {})),
        "n_prediction_rows": int(len(predictions)),
        "backtest_n_jobs": int(config.get("backtest", {}).get("n_jobs", 1)),
        "training_diagnostics": training_diagnostics_metadata or {"enabled": False},
    }


def _feature_extraction_metadata(config: dict[str, Any], predictions: pd.DataFrame) -> dict[str, Any]:
    cfg = config.get("feature_extraction", {})
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "method": cfg.get("method", "none"),
        "apply_to_models": list(cfg.get("apply_to_models", [])),
        "keep_original_features": bool(cfg.get("keep_original_features", False)),
        "n_prediction_rows": int(len(predictions)),
        "methods_in_predictions": sorted(predictions["feature_extraction_method"].dropna().unique().tolist())
        if "feature_extraction_method" in predictions.columns and not predictions.empty
        else [],
    }


def _gap_metrics(predictions: pd.DataFrame, target_columns: list[str]) -> dict[str, Any]:
    if predictions.empty:
        return {"metrics_by_gap_day": {}, "metrics_by_release_gap_size": {}}
    return {
        "metrics_by_gap_day": {
            str(gap_day): evaluate_prediction_groups(group, target_columns)
            for gap_day, group in predictions.groupby("gap_day")
        },
        "metrics_by_release_gap_size": {
            str(size): evaluate_prediction_groups(group, target_columns)
            for size, group in predictions.groupby("release_gap_size")
        },
    }


def _filter_prediction_frame(predictions: pd.DataFrame, selected_target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    if predictions.empty or len(selected_target_dates) == 0:
        return predictions.iloc[0:0].copy()
    selected = set(pd.DatetimeIndex(selected_target_dates).date)
    dates = pd.to_datetime(predictions["target_date"]).dt.date
    return predictions.loc[dates.isin(selected)].reset_index(drop=True)


def _model_implied_minus_official_series(predictions: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    base_columns = [
        "date",
        "cutoff_date",
        "target_date",
        "gap_day",
        "release_gap_size",
        "model_type",
    ]
    residual_columns = [f"model_implied_minus_official_{target}" for target in target_columns]
    columns = base_columns + residual_columns
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    result = predictions.copy()
    for target in target_columns:
        result[f"model_implied_minus_official_{target}"] = (
            result[f"pred_{target}"].astype(float) - result[f"actual_{target}"].astype(float)
        )
    for column in base_columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result[columns].reset_index(drop=True)
