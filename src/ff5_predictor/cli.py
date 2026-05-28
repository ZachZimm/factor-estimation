from __future__ import annotations

import argparse
import json
import logging
import webbrowser
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from ff5_predictor.config import load_config
from ff5_predictor.data_famafrench import load_ff5
from ff5_predictor.data_yfinance import load_market_data
from ff5_predictor.dataset import build_modeling_dataset, split_feature_target_columns
from ff5_predictor.evaluation import compare_against_baseline, evaluate_predictions
from ff5_predictor.evaluation import evaluate_prediction_groups
from ff5_predictor.experiment_config import AVAILABLE_MODELS, VISIBLE_MODELS
from ff5_predictor.experiment_io import write_metrics, write_predictions
from ff5_predictor.experiment_runner import run_experiment, train_one_model
from ff5_predictor.factor_viewer import write_factor_viewer_html
from ff5_predictor.io import ensure_dir, package_version, read_parquet_with_metadata, utc_now_iso, write_parquet_with_metadata
from ff5_predictor.nowcast_dataset import build_nowcast_dataset
from ff5_predictor.nowcast_io import create_nowcast_run_dir, write_json, write_nowcast_dataset, write_yaml
from ff5_predictor.production_nowcast import run_production_nowcast
from ff5_predictor.release_gap_backtest import run_release_gap_backtest
from ff5_predictor.rolling_train import naive_previous_value_baseline, rolling_predict

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def modeling_dataset_path(config: dict) -> Path:
    return Path(config["data"].get("processed_dir", "data/processed")) / "modeling_dataset.parquet"


def cmd_download_data(config: dict) -> None:
    load_ff5(config)
    load_market_data(config)
    LOGGER.info("Downloaded or loaded cached source data")


def cmd_build_dataset(config: dict) -> pd.DataFrame:
    ff5_df = load_ff5(config)
    market_df = load_market_data(config)
    modeling_df = build_modeling_dataset(ff5_df, market_df, config)
    target_columns = config["prediction"]["target_columns"]
    feature_columns, _ = split_feature_target_columns(modeling_df, target_columns)
    path = modeling_dataset_path(config)
    metadata = {
        "source": "ff5_plus_yfinance_features",
        "dataset": "modeling_dataset",
        "download_timestamp_utc": utc_now_iso(),
        "start_date": str(modeling_df.index.min().date()) if not modeling_df.empty else None,
        "end_date": str(modeling_df.index.max().date()) if not modeling_df.empty else None,
        "tickers": config["data"]["tickers"],
        "library_versions": {
            "pandas": package_version("pandas"),
        },
        "units": "decimal_targets",
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "horizon": config["prediction"].get("horizon", 1),
    }
    write_parquet_with_metadata(modeling_df, path, metadata)
    LOGGER.info("Wrote modeling dataset to %s", path)
    return modeling_df


def load_or_build_dataset(config: dict) -> pd.DataFrame:
    path = modeling_dataset_path(config)
    if path.exists() and not bool(config["data"].get("force_refresh", False)):
        df, _ = read_parquet_with_metadata(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if not df.empty:
            return df
        LOGGER.warning("Cached modeling dataset at %s is empty; rebuilding it", path)
    return cmd_build_dataset(config)


def cmd_train_rolling(config: dict) -> pd.DataFrame:
    modeling_df = load_or_build_dataset(config)
    if modeling_df.empty:
        raise ValueError(
            "Modeling dataset is empty after feature/target alignment. "
            "Check downloaded market data, FF5 date coverage, and feature windows."
        )
    target_columns = list(config["prediction"]["target_columns"])
    feature_columns, target_columns = split_feature_target_columns(modeling_df, target_columns)
    model_predictions = rolling_predict(modeling_df, feature_columns, target_columns, config)
    baseline_predictions = naive_previous_value_baseline(
        modeling_df,
        target_columns,
        step_size=int(config["training"].get("step_size", 1)),
    )
    predictions = pd.concat([model_predictions, baseline_predictions], ignore_index=True)
    output_path = Path(config["output"]["predictions_path"])
    ensure_dir(output_path.parent)
    predictions.to_csv(output_path, index=False)
    LOGGER.info("Wrote predictions to %s", output_path)
    return predictions


def cmd_evaluate(config: dict) -> dict:
    predictions_path = Path(config["output"]["predictions_path"])
    if not predictions_path.exists():
        cmd_train_rolling(config)
    try:
        predictions = pd.read_csv(predictions_path)
    except EmptyDataError:
        LOGGER.warning("Predictions file at %s is empty; regenerating it", predictions_path)
        predictions = cmd_train_rolling(config)
    if predictions.empty:
        raise ValueError(
            "Predictions file has headers but no rows. Lower min_train_rows, "
            "increase source data coverage, or inspect the modeling dataset."
        )
    target_columns = list(config["prediction"]["target_columns"])
    metrics: dict = {"models": {}, "baseline_comparison": {}}
    for model_type, group in predictions.groupby("model_type"):
        metrics["models"][model_type] = evaluate_predictions(group, target_columns)

    if "ridge" in metrics["models"] and "naive_previous" in metrics["models"]:
        model_group = predictions[predictions["model_type"] == "ridge"]
        baseline_group = predictions[predictions["model_type"] == "naive_previous"]
        metrics["baseline_comparison"]["naive_previous"] = compare_against_baseline(
            model_group,
            baseline_group,
            target_columns,
        )

    metrics_path = Path(config["output"]["metrics_path"])
    ensure_dir(metrics_path.parent)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    LOGGER.info("Wrote metrics to %s", metrics_path)
    return metrics


def cmd_run_all(config: dict) -> None:
    cmd_download_data(config)
    cmd_build_dataset(config)
    cmd_train_rolling(config)
    cmd_evaluate(config)


def cmd_run_experiment(config: dict) -> None:
    run_experiment(config)


def cmd_nowcast(config: dict) -> None:
    result = run_production_nowcast(config)
    LOGGER.info("Wrote nowcast outputs to %s", result.run_dir)


def cmd_backtest_nowcast(config: dict) -> None:
    result = run_release_gap_backtest(config)
    LOGGER.info("Wrote nowcast backtest outputs to %s", result.run_dir)


def cmd_build_nowcast_dataset(config: dict) -> None:
    ff5_df = load_ff5(config)
    market_df = load_market_data(config)
    dataset = build_nowcast_dataset(ff5_df, market_df, config)
    run_dir = create_nowcast_run_dir(config)
    write_yaml(run_dir / "config_resolved.yaml", config)
    write_nowcast_dataset(run_dir, "train.parquet", dataset.train_df)
    write_nowcast_dataset(run_dir, "inference.parquet", dataset.inference_df)
    write_json(run_dir / "metadata" / "dataset_metadata.json", dataset.metadata)
    LOGGER.info("Wrote nowcast dataset outputs to %s", run_dir)


def cmd_train_model(config: dict, model_type: str) -> None:
    modeling_df = cmd_build_dataset(config)
    target_columns = list(config["prediction"]["target_columns"])
    feature_columns, target_columns = split_feature_target_columns(modeling_df, target_columns)
    predictions, _ = train_one_model(model_type, modeling_df, feature_columns, target_columns, config)
    write_predictions(config, model_type, predictions)
    metrics = evaluate_prediction_groups(predictions, target_columns)
    write_metrics(config, f"{model_type}_metrics.json", metrics)
    LOGGER.info("Wrote experiment predictions and metrics for %s", model_type)


def cmd_list_models(include_hidden: bool = False) -> None:
    for model in (AVAILABLE_MODELS if include_hidden else VISIBLE_MODELS):
        print(model)


def cmd_torch_info(config: dict) -> None:
    import torch

    from ff5_predictor.torch_models.common import select_device

    device = select_device(config)
    print(f"torch: {torch.__version__}")
    print(f"torch_cuda_build: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"selected_device: {device}")
    if device.type == "cuda":
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        print(f"cuda_device_name: {torch.cuda.get_device_name(device)}")


def cmd_view_factors(
    config: dict,
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str | None = None,
    open_browser: bool = False,
) -> Path:
    ff5_df = load_ff5(config)
    path = Path(output_path or "data/processed/ff5_factor_viewer.html")
    written = write_factor_viewer_html(ff5_df, path, start_date=start_date, end_date=end_date)
    LOGGER.info("Wrote FF5 factor viewer to %s", written)
    if open_browser:
        webbrowser.open(written.resolve().as_uri())
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ff5-predictor")
    parser.add_argument(
        "command",
        choices=[
            "download-data",
            "build-dataset",
            "train-rolling",
            "evaluate",
            "run-all",
            "run-experiment",
            "train-model",
            "list-models",
            "torch-info",
            "view-factors",
            "nowcast",
            "backtest-nowcast",
            "build-nowcast-dataset",
        ],
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", help="Model type for train-model")
    parser.add_argument("--include-hidden", action="store_true", help="Include deprecated hidden models in list-models")
    parser.add_argument("--start-date", help="Start date for view-factors, YYYY-MM-DD")
    parser.add_argument("--end-date", help="End date for view-factors, YYYY-MM-DD")
    parser.add_argument(
        "--viewer-output",
        default="data/processed/ff5_factor_viewer.html",
        help="HTML output path for view-factors",
    )
    parser.add_argument("--open-browser", action="store_true", help="Open generated factor viewer in the browser")
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "download-data":
        cmd_download_data(config)
    elif args.command == "build-dataset":
        cmd_build_dataset(config)
    elif args.command == "train-rolling":
        cmd_train_rolling(config)
    elif args.command == "evaluate":
        cmd_evaluate(config)
    elif args.command == "run-all":
        cmd_run_all(config)
    elif args.command == "run-experiment":
        cmd_run_experiment(config)
    elif args.command == "nowcast":
        cmd_nowcast(config)
    elif args.command == "backtest-nowcast":
        cmd_backtest_nowcast(config)
    elif args.command == "build-nowcast-dataset":
        cmd_build_nowcast_dataset(config)
    elif args.command == "train-model":
        if not args.model:
            parser.error("--model is required for train-model")
        cmd_train_model(config, args.model)
    elif args.command == "list-models":
        cmd_list_models(include_hidden=args.include_hidden)
    elif args.command == "torch-info":
        cmd_torch_info(config)
    elif args.command == "view-factors":
        cmd_view_factors(
            config,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.viewer_output,
            open_browser=args.open_browser,
        )


if __name__ == "__main__":
    main()
