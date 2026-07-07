from __future__ import annotations

from pathlib import Path

import pandas as pd

from ff5_predictor.cli import build_parser
from ff5_predictor.performance_analysis import (
    add_bps_columns,
    find_latest_architecture_manifest,
    load_architecture_manifest,
    run_performance_analysis,
)


TARGETS = ["Mkt-RF", "SMB"]


def _prediction_rows(model_type: str, dates: pd.DatetimeIndex, error_scale: float) -> list[dict]:
    rows = []
    for idx, date in enumerate(dates):
        actual_mkt = 0.001 * (idx + 1)
        actual_smb = -0.0005 * (idx + 1)
        rows.append(
            {
                "date": date.date().isoformat(),
                "target_date": date.date().isoformat(),
                "cutoff_date": (date - pd.Timedelta(days=1)).date().isoformat(),
                "gap_day": idx % 3 + 1,
                "release_gap_size": 5,
                "model_type": model_type,
                "pred_Mkt-RF": actual_mkt - error_scale,
                "actual_Mkt-RF": actual_mkt,
                "pred_SMB": actual_smb + error_scale * 0.5,
                "actual_SMB": actual_smb,
            }
        )
    return rows


def _write_run(root: Path, run_name: str, stamp: str, models: dict[str, float], *, training: bool = False) -> Path:
    run_dir = root / run_name / stamp
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    rows = []
    for model_type, error in models.items():
        rows.extend(_prediction_rows(model_type, dates, error))
    pd.DataFrame(rows).to_csv(predictions_dir / "release_gap_predictions.csv", index=False)
    (run_dir / "metrics").mkdir()
    pd.DataFrame(
        {
            "model_type": list(models),
            "avg_rmse": [error for error in models.values()],
            "avg_mae": [error * 0.8 for error in models.values()],
        }
    ).to_csv(run_dir / "metrics" / "model_ranking.csv", index=False)
    if training:
        training_dir = run_dir / "training"
        training_dir.mkdir()
        history = pd.DataFrame(
            [
                {
                    "model_type": "tft",
                    "cutoff_date": "2024-01-31",
                    "latest_market_date": "2024-02-05",
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "validation_loss": 1.2 / epoch,
                    "train_rmse": 0.004 / epoch,
                    "validation_rmse": 0.005 / epoch,
                    "train_directional_accuracy": 0.55,
                    "validation_directional_accuracy": 0.52,
                    "device": "cpu",
                    "feature_extraction_method": "group_pca",
                    "lookback_rows": 252,
                    "elapsed_seconds": 0.1,
                    "is_best_epoch": epoch == 2,
                }
                for epoch in [1, 2]
            ]
        )
        history.to_csv(training_dir / "neural_training_history.csv", index=False)
        (training_dir / "tft_training_curves.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><text>tft training curves</text></svg>',
            encoding="utf-8",
        )
    return run_dir


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_manifest_filters_completed_runs_and_latest_selection(tmp_path: Path) -> None:
    root = tmp_path / "architecture_comparison_runs"
    run_dir = _write_run(tmp_path / "nowcasts", "linear", "20240101_000000", {"ridge": 0.001})
    old_manifest = root / "20240101_000000" / "manifest.tsv"
    new_manifest = root / "20240201_000000" / "manifest.tsv"
    _write_manifest(
        old_manifest,
        [
            {
                "started_at_utc": "2024-01-01T00:00:00Z",
                "label": "old",
                "config": "config.yaml",
                "run_name": "linear",
                "latest_output_dir": str(run_dir),
                "status": "completed",
            }
        ],
    )
    _write_manifest(
        new_manifest,
        [
            {
                "started_at_utc": "2024-02-01T00:00:00Z",
                "label": "started only",
                "config": "config.yaml",
                "run_name": "linear",
                "latest_output_dir": "",
                "status": "started",
            },
            {
                "started_at_utc": "2024-02-01T00:01:00Z",
                "label": "new",
                "config": "config.yaml",
                "run_name": "linear",
                "latest_output_dir": str(run_dir),
                "status": "completed",
            },
        ],
    )

    assert find_latest_architecture_manifest(root) == new_manifest
    completed = load_architecture_manifest(new_manifest)
    assert len(completed) == 1
    assert completed.iloc[0]["label"] == "new"


def test_add_bps_columns() -> None:
    ranking = add_bps_columns(pd.DataFrame({"model_type": ["ridge"], "avg_rmse": [0.0025], "avg_mae": [0.001]}))
    assert float(ranking.loc[0, "avg_rmse_bps"]) == 25.0
    assert float(ranking.loc[0, "avg_mae_bps"]) == 10.0


def test_cli_parses_analyze_performance_args() -> None:
    args = build_parser().parse_args(
        [
            "analyze-performance",
            "--manifest",
            "manifest.tsv",
            "--output-dir",
            "out",
            "--title",
            "Report",
        ]
    )

    assert args.command == "analyze-performance"
    assert args.manifest == "manifest.tsv"
    assert args.output_dir == "out"
    assert args.title == "Report"


def test_performance_analysis_writes_report_tables_figures_and_training(tmp_path: Path) -> None:
    output_root = tmp_path / "nowcasts"
    linear_dir = _write_run(
        output_root,
        "linear",
        "20240101_000000",
        {"rolling_mean": 0.003, "ewma": 0.004, "ridge": 0.0015},
    )
    tft_dir = _write_run(
        output_root,
        "tft",
        "20240101_010000",
        {"rolling_mean": 0.003, "ewma": 0.004, "tft": 0.0025},
        training=True,
    )
    manifest = tmp_path / "architecture_comparison_runs" / "20240101_000000" / "manifest.tsv"
    _write_manifest(
        manifest,
        [
            {
                "started_at_utc": "2024-01-01T00:00:00Z",
                "label": "linear",
                "config": "linear.yaml",
                "run_name": "linear",
                "latest_output_dir": str(linear_dir),
                "status": "completed",
            },
            {
                "started_at_utc": "2024-01-01T01:00:00Z",
                "label": "tft",
                "config": "tft.yaml",
                "run_name": "tft",
                "latest_output_dir": str(tft_dir),
                "status": "completed",
            },
        ],
    )

    result = run_performance_analysis(manifest, output_dir=tmp_path / "report", title="Synthetic Performance")

    assert result.html_path.exists()
    html = result.html_path.read_text(encoding="utf-8")
    assert "Architecture ranking" in html
    assert "Factor-level performance" in html
    assert "Release-gap behavior" in html
    assert "Neural training diagnostics" in html
    assert "tft training curves" in html
    assert (result.output_dir / "tables" / "per_factor_metrics.csv").exists()
    assert (result.output_dir / "figures" / "factor_rmse_heatmap.svg").exists()
    assert "ridge" in set(result.ranking["model_type"])
    combined = pd.read_csv(result.output_dir / "tables" / "combined_predictions.csv")
    assert len(combined.loc[combined["model_type"] == "ewma"]) == 30
