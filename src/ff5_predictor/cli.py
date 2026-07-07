from __future__ import annotations

import argparse
from copy import deepcopy
import logging
import webbrowser
from pathlib import Path

from ff5_predictor.config import load_config
from ff5_predictor.data_famafrench import load_ff5
from ff5_predictor.data_yfinance import load_market_data
from ff5_predictor.elasticnet_validation import run_elasticnet_validation
from ff5_predictor.experiment_config import AVAILABLE_MODELS
from ff5_predictor.factor_viewer import (
    load_predictions_csv,
    resolve_predictions_path,
    write_factor_viewer_html,
)
from ff5_predictor.nowcast_dataset import build_nowcast_dataset
from ff5_predictor.nowcast_io import create_nowcast_run_dir, write_json, write_nowcast_dataset, write_yaml
from ff5_predictor.latest_nowcast import run_latest_nowcast
from ff5_predictor.model_implied_series import run_model_implied_series
from ff5_predictor.performance_analysis import run_performance_analysis
from ff5_predictor.release_gap_backtest import run_release_gap_backtest
from ff5_predictor.residual_analysis import run_residual_analysis

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def cmd_download_data(config: dict) -> None:
    load_ff5(config)
    load_market_data(config)
    LOGGER.info("Downloaded or loaded cached source data")


def config_with_date_filter(config: dict, start_date: str | None, end_date: str | None) -> dict:
    if not start_date and not end_date:
        return config
    filtered = deepcopy(config)
    filtered["date_filter"] = {
        "start_date": start_date,
        "end_date": end_date,
    }
    return filtered


def cmd_nowcast(config: dict) -> None:
    result = run_latest_nowcast(config)
    LOGGER.info("Wrote nowcast outputs to %s", result.run_dir)


def cmd_backtest_nowcast(config: dict) -> None:
    result = run_release_gap_backtest(config)
    LOGGER.info("Wrote nowcast backtest outputs to %s", result.run_dir)


def cmd_model_implied_series(config: dict) -> None:
    result = run_model_implied_series(config)
    LOGGER.info("Wrote model-implied FF5 series outputs to %s", result.run_dir)


def cmd_analyze_residuals(
    config: dict,
    predictions_csv: str | None = None,
    run_dir: str | None = None,
    output_dir: str | None = None,
    model_type: str | None = None,
    release_gap_size: int | None = None,
    gap_day: int | None = None,
) -> None:
    result = run_residual_analysis(
        config,
        predictions_csv=predictions_csv,
        run_dir=run_dir,
        output_dir=output_dir,
        model_type=model_type,
        release_gap_size=release_gap_size,
        gap_day=gap_day,
    )
    LOGGER.info("Wrote residual analysis outputs to %s", result.run_dir)


def cmd_analyze_performance(
    manifest_path: str | None = None,
    output_dir: str | None = None,
    title: str | None = None,
    open_browser: bool = False,
) -> None:
    result = run_performance_analysis(
        manifest_path,
        output_dir=output_dir,
        title=title,
        open_browser=open_browser,
    )
    LOGGER.info("Wrote performance analysis report to %s", result.html_path)


def cmd_validate_elasticnet(config: dict) -> None:
    result = run_elasticnet_validation(config)
    LOGGER.info("Wrote ElasticNet validation outputs to %s", result.run_dir)


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


def cmd_list_models() -> None:
    for model in AVAILABLE_MODELS:
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
    predictions_csv: str | None = None,
    run_dir: str | None = None,
    model_type: str | None = None,
    gap_day: int | None = None,
) -> Path:
    ff5_df = load_ff5(config)
    path = Path(output_path or "data/processed/ff5_factor_viewer.html")
    target_columns = list(config["prediction"]["target_columns"])
    predictions_path = resolve_predictions_path(predictions_csv=predictions_csv, run_dir=run_dir)
    predictions_df = load_predictions_csv(predictions_path) if predictions_path is not None else None
    run_label = str(predictions_path) if predictions_path is not None else None
    written = write_factor_viewer_html(
        ff5_df,
        path,
        start_date=start_date,
        end_date=end_date,
        predictions_df=predictions_df,
        target_columns=target_columns,
        default_model=model_type,
        default_gap_day=gap_day,
        run_label=run_label,
    )
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
            "list-models",
            "torch-info",
            "view-factors",
            "nowcast",
            "backtest-nowcast",
            "model-implied-series",
            "analyze-residuals",
            "analyze-performance",
            "validate-elasticnet",
            "build-nowcast-dataset",
        ],
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start-date", help="Start date for prediction or view filtering, YYYY-MM-DD")
    parser.add_argument("--end-date", help="End date for prediction or view filtering, YYYY-MM-DD")
    parser.add_argument(
        "--viewer-output",
        default="data/processed/ff5_factor_viewer.html",
        help="HTML output path for view-factors",
    )
    parser.add_argument("--open-browser", action="store_true", help="Open generated factor viewer in the browser")
    parser.add_argument(
        "--predictions-csv",
        help="Prediction CSV from a nowcast/backtest run to overlay in view-factors",
    )
    parser.add_argument(
        "--run-dir",
        help="Nowcast run directory; auto-selects predictions/release_gap_predictions.csv or latest_nowcast.csv",
    )
    parser.add_argument("--model-type", help="Default model to show when comparing run predictions")
    parser.add_argument("--gap-day", type=int, help="Default release-gap day filter for backtest prediction rows")
    parser.add_argument("--release-gap-size", type=int, help="Release-gap size filter for residual analysis")
    parser.add_argument("--analysis-output-dir", help="Output directory for analyze-residuals")
    parser.add_argument("--manifest", help="Architecture comparison manifest for analyze-performance")
    parser.add_argument("--output-dir", help="Output directory for analyze-performance")
    parser.add_argument("--title", help="Report title for analyze-performance")
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "download-data":
        cmd_download_data(config)
    elif args.command == "nowcast":
        cmd_nowcast(config_with_date_filter(config, args.start_date, args.end_date))
    elif args.command == "backtest-nowcast":
        cmd_backtest_nowcast(config_with_date_filter(config, args.start_date, args.end_date))
    elif args.command == "model-implied-series":
        cmd_model_implied_series(config_with_date_filter(config, args.start_date, args.end_date))
    elif args.command == "analyze-residuals":
        cmd_analyze_residuals(
            config_with_date_filter(config, args.start_date, args.end_date),
            predictions_csv=args.predictions_csv,
            run_dir=args.run_dir,
            output_dir=args.analysis_output_dir,
            model_type=args.model_type,
            release_gap_size=args.release_gap_size,
            gap_day=args.gap_day,
        )
    elif args.command == "analyze-performance":
        cmd_analyze_performance(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            title=args.title,
            open_browser=args.open_browser,
        )
    elif args.command == "validate-elasticnet":
        cmd_validate_elasticnet(config_with_date_filter(config, args.start_date, args.end_date))
    elif args.command == "build-nowcast-dataset":
        cmd_build_nowcast_dataset(config_with_date_filter(config, args.start_date, args.end_date))
    elif args.command == "list-models":
        cmd_list_models()
    elif args.command == "torch-info":
        cmd_torch_info(config)
    elif args.command == "view-factors":
        cmd_view_factors(
            config,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.viewer_output,
            open_browser=args.open_browser,
            predictions_csv=args.predictions_csv,
            run_dir=args.run_dir,
            model_type=args.model_type,
            gap_day=args.gap_day,
        )


if __name__ == "__main__":
    main()
