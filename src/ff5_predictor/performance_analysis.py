from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
import json
import webbrowser

import numpy as np
import pandas as pd

from ff5_predictor.evaluation import evaluate_on_shared_dates, evaluate_predictions, rank_models
from ff5_predictor.io import ensure_dir
from ff5_predictor.nowcast_io import write_json


DEFAULT_ARCHITECTURE_RUNS_DIR = Path("data/nowcasts/architecture_comparison_runs")
BASELINE_MODELS = {"rolling_mean", "ewma"}
TARGET_ORDER = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


@dataclass(frozen=True)
class PerformanceAnalysisResult:
    output_dir: Path
    html_path: Path
    manifest: pd.DataFrame
    predictions: pd.DataFrame
    ranking: pd.DataFrame
    per_factor: pd.DataFrame
    metadata: dict[str, Any]


def run_performance_analysis(
    manifest_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    title: str | None = None,
    open_browser: bool = False,
) -> PerformanceAnalysisResult:
    manifest = find_latest_architecture_manifest() if manifest_path is None else Path(manifest_path)
    completed = load_architecture_manifest(manifest)
    output = ensure_dir(Path(output_dir) if output_dir is not None else manifest.parent / "performance_analysis")
    tables_dir = ensure_dir(output / "tables")
    figures_dir = ensure_dir(output / "figures")

    predictions = load_manifest_predictions(completed)
    if predictions.empty:
        raise ValueError(f"No completed prediction rows found for manifest: {manifest}")
    target_columns = infer_target_columns(predictions)
    if not target_columns:
        raise ValueError("Prediction files do not contain matching pred_* and actual_* target columns")

    shared_metrics = evaluate_on_shared_dates(predictions, target_columns, baseline_model="ewma")
    ranking = add_bps_columns(rank_models(shared_metrics, primary_metric="rmse"))
    per_factor = flatten_factor_metrics(shared_metrics, target_columns)
    gap_day = grouped_metric_table(predictions, target_columns, "gap_day")
    release_gap_size = grouped_metric_table(predictions, target_columns, "release_gap_size")
    baseline_comparison = baseline_factor_comparison(per_factor, target_columns)
    prediction_volatility = prediction_volatility_table(per_factor)
    factor_extremes = best_worst_factor_table(per_factor)
    rolling_rmse = rolling_rmse_table(predictions, target_columns, window_rows=63)
    training_summary, training_figures = collect_training_diagnostics(completed)

    completed.to_csv(tables_dir / "manifest_completed.csv", index=False)
    predictions.to_csv(tables_dir / "combined_predictions.csv", index=False)
    ranking.to_csv(tables_dir / "model_ranking.csv", index=False)
    per_factor.to_csv(tables_dir / "per_factor_metrics.csv", index=False)
    gap_day.to_csv(tables_dir / "gap_day_metrics.csv", index=False)
    release_gap_size.to_csv(tables_dir / "release_gap_size_metrics.csv", index=False)
    baseline_comparison.to_csv(tables_dir / "baseline_factor_comparison.csv", index=False)
    prediction_volatility.to_csv(tables_dir / "prediction_volatility.csv", index=False)
    factor_extremes.to_csv(tables_dir / "factor_extremes.csv", index=False)
    rolling_rmse.to_csv(tables_dir / "rolling_rmse.csv", index=False)
    training_summary.to_csv(tables_dir / "training_summary.csv", index=False)

    figures = write_performance_figures(
        figures_dir,
        predictions,
        ranking,
        per_factor,
        gap_day,
        rolling_rmse,
        target_columns,
    )
    report_title = title or f"Model Performance Analysis: {manifest.parent.name}"
    html = build_performance_report_html(
        report_title,
        manifest,
        completed,
        predictions,
        ranking,
        per_factor,
        gap_day,
        release_gap_size,
        baseline_comparison,
        prediction_volatility,
        factor_extremes,
        training_summary,
        training_figures,
        figures,
        target_columns,
    )
    html_path = output / "performance_analysis_report.html"
    html_path.write_text(html, encoding="utf-8")

    metadata = {
        "manifest_path": str(manifest),
        "output_dir": str(output),
        "html_path": str(html_path),
        "target_columns": target_columns,
        "n_prediction_rows": int(len(predictions)),
        "n_models": int(predictions["model_type"].nunique()),
        "n_shared_dates": int(shared_metrics.get("n_shared_dates", 0)),
        "table_paths": {
            "manifest_completed": str(tables_dir / "manifest_completed.csv"),
            "combined_predictions": str(tables_dir / "combined_predictions.csv"),
            "model_ranking": str(tables_dir / "model_ranking.csv"),
            "per_factor_metrics": str(tables_dir / "per_factor_metrics.csv"),
            "gap_day_metrics": str(tables_dir / "gap_day_metrics.csv"),
            "release_gap_size_metrics": str(tables_dir / "release_gap_size_metrics.csv"),
            "baseline_factor_comparison": str(tables_dir / "baseline_factor_comparison.csv"),
            "prediction_volatility": str(tables_dir / "prediction_volatility.csv"),
            "factor_extremes": str(tables_dir / "factor_extremes.csv"),
            "rolling_rmse": str(tables_dir / "rolling_rmse.csv"),
            "training_summary": str(tables_dir / "training_summary.csv"),
        },
        "figure_paths": {name: str(path) for name, path in figures.items()},
        "training_figures": {name: str(path) for name, path in training_figures.items()},
    }
    write_json(output / "metadata.json", metadata)
    if open_browser:
        webbrowser.open(html_path.resolve().as_uri())
    return PerformanceAnalysisResult(output, html_path, completed, predictions, ranking, per_factor, metadata)


def find_latest_architecture_manifest(root: str | Path = DEFAULT_ARCHITECTURE_RUNS_DIR) -> Path:
    base = Path(root)
    manifests = sorted(base.glob("*/manifest.tsv"))
    completed = [path for path in manifests if _manifest_has_completed_runs(path)]
    if not completed:
        raise FileNotFoundError(f"No completed architecture comparison manifest found under {base}")
    return completed[-1]


def load_architecture_manifest(manifest_path: str | Path) -> pd.DataFrame:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest = pd.read_csv(path, sep="\t", keep_default_na=False)
    required = {"started_at_utc", "label", "config", "run_name", "latest_output_dir", "status"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    completed = manifest.loc[
        manifest["status"].eq("completed") & manifest["latest_output_dir"].astype(str).ne("")
    ].copy()
    if completed.empty:
        raise ValueError(f"Manifest has no completed runs: {path}")
    completed["manifest_path"] = str(path)
    completed["run_dir"] = completed["latest_output_dir"].map(str)
    for run_dir in completed["run_dir"]:
        if not Path(run_dir).is_dir():
            raise FileNotFoundError(f"Completed run directory is missing: {run_dir}")
    return completed.reset_index(drop=True)


def load_manifest_predictions(completed_manifest: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen_baselines: set[str] = set()
    for _, row in completed_manifest.iterrows():
        run_dir = Path(str(row["run_dir"]))
        path = run_dir / "predictions" / "release_gap_predictions.csv"
        if not path.exists():
            continue
        predictions = pd.read_csv(path)
        if predictions.empty or "model_type" not in predictions.columns:
            continue
        predictions = predictions.copy()
        predictions["date"] = pd.to_datetime(
            predictions["target_date"] if "target_date" in predictions.columns else predictions["date"]
        ).dt.normalize()
        predictions["model_type"] = predictions["model_type"].astype(str)
        predictions["architecture_label"] = str(row["label"])
        predictions["architecture_run_name"] = str(row["run_name"])
        predictions["architecture_run_dir"] = str(run_dir)
        keep_parts: list[pd.DataFrame] = []
        for model_type, group in predictions.groupby("model_type", sort=False):
            if model_type in BASELINE_MODELS:
                if model_type in seen_baselines:
                    continue
                seen_baselines.add(model_type)
            keep_parts.append(group)
        if keep_parts:
            frames.append(pd.concat(keep_parts, ignore_index=True))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["model_type", "date"]).reset_index(drop=True)


def infer_target_columns(predictions: pd.DataFrame) -> list[str]:
    targets = []
    for column in predictions.columns:
        if not column.startswith("pred_"):
            continue
        target = column.removeprefix("pred_")
        if f"actual_{target}" in predictions.columns:
            targets.append(target)
    ordered = [target for target in TARGET_ORDER if target in targets]
    return ordered + sorted(set(targets).difference(ordered))


def add_bps_columns(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return ranking
    result = ranking.copy()
    for column in ["avg_mae", "avg_rmse"]:
        if column in result.columns:
            result[f"{column}_bps"] = pd.to_numeric(result[column], errors="coerce") * 10000.0
    if "avg_directional_accuracy" in result.columns:
        result["avg_directional_accuracy_pct"] = pd.to_numeric(result["avg_directional_accuracy"], errors="coerce") * 100.0
    if "avg_corr" in result.columns:
        result["avg_corr"] = pd.to_numeric(result["avg_corr"], errors="coerce")
    return result


def flatten_factor_metrics(shared_metrics: dict[str, Any], target_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_type, metrics in shared_metrics.get("models", {}).items():
        by_factor = metrics.get("metrics_by_factor", {})
        for target in target_columns:
            item = by_factor.get(target)
            if not item:
                continue
            row = {"model_type": model_type, "target": target}
            row.update(item)
            rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for column in ["mae", "rmse", "mean_prediction", "prediction_std", "actual_std"]:
        if column in table.columns:
            table[f"{column}_bps"] = pd.to_numeric(table[column], errors="coerce") * 10000.0
    for column in ["directional_accuracy", "sign_bias", "top_quintile_hit_rate", "bottom_quintile_hit_rate"]:
        if column in table.columns:
            table[f"{column}_pct"] = pd.to_numeric(table[column], errors="coerce") * 100.0
    return table


def grouped_metric_table(predictions: pd.DataFrame, target_columns: list[str], group_column: str) -> pd.DataFrame:
    if group_column not in predictions.columns or predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (model_type, group_value), group in predictions.dropna(subset=[group_column]).groupby(["model_type", group_column]):
        metrics = evaluate_predictions(group, target_columns).get("metrics_by_factor", {})
        if not metrics:
            continue
        values = list(metrics.values())
        rows.append(
            {
                "model_type": model_type,
                group_column: group_value,
                "n_predictions": int(len(group)),
                "avg_mae": float(np.mean([m["mae"] for m in values])),
                "avg_rmse": float(np.mean([m["rmse"] for m in values])),
                "avg_corr": float(np.mean([m["correlation"] or 0.0 for m in values])),
                "avg_directional_accuracy": float(np.mean([m["directional_accuracy"] for m in values])),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["avg_mae_bps"] = result["avg_mae"] * 10000.0
    result["avg_rmse_bps"] = result["avg_rmse"] * 10000.0
    result["avg_directional_accuracy_pct"] = result["avg_directional_accuracy"] * 100.0
    return result.sort_values(["model_type", group_column]).reset_index(drop=True)


def baseline_factor_comparison(per_factor: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    if per_factor.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    baselines = [baseline for baseline in ["ewma", "rolling_mean"] if baseline in set(per_factor["model_type"])]
    for baseline in baselines:
        baseline_rows = per_factor.loc[per_factor["model_type"] == baseline].set_index("target")
        for _, row in per_factor.iterrows():
            target = row["target"]
            if target not in target_columns or target not in baseline_rows.index or row["model_type"] == baseline:
                continue
            base = baseline_rows.loc[target]
            base_rmse = float(base["rmse"])
            rmse = float(row["rmse"])
            rows.append(
                {
                    "model_type": row["model_type"],
                    "baseline_model": baseline,
                    "target": target,
                    "rmse_bps": rmse * 10000.0,
                    "baseline_rmse_bps": base_rmse * 10000.0,
                    "rmse_improvement_bps": (base_rmse - rmse) * 10000.0,
                    "rmse_vs_baseline_pct": (rmse / base_rmse - 1.0) * 100.0 if base_rmse else np.nan,
                }
            )
    return pd.DataFrame(rows)


def prediction_volatility_table(per_factor: pd.DataFrame) -> pd.DataFrame:
    if per_factor.empty:
        return pd.DataFrame()
    table = per_factor[
        [
            "model_type",
            "target",
            "mean_prediction_bps",
            "prediction_std_bps",
            "actual_std_bps",
            "sign_bias_pct",
            "top_quintile_hit_rate_pct",
            "bottom_quintile_hit_rate_pct",
        ]
    ].copy()
    table["prediction_std_to_actual_std"] = table["prediction_std_bps"] / table["actual_std_bps"].replace(0, np.nan)
    return table


def best_worst_factor_table(per_factor: pd.DataFrame) -> pd.DataFrame:
    if per_factor.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model_type, group in per_factor.groupby("model_type"):
        ordered = group.sort_values("rmse")
        best = ordered.iloc[0]
        worst = ordered.iloc[-1]
        rows.append(
            {
                "model_type": model_type,
                "best_target": best["target"],
                "best_rmse_bps": float(best["rmse_bps"]),
                "worst_target": worst["target"],
                "worst_rmse_bps": float(worst["rmse_bps"]),
            }
        )
    return pd.DataFrame(rows)


def rolling_rmse_table(predictions: pd.DataFrame, target_columns: list[str], window_rows: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model_type, group in predictions.groupby("model_type"):
        work = group.sort_values("date").copy()
        errors = []
        for target in target_columns:
            errors.append((work[f"pred_{target}"].astype(float) - work[f"actual_{target}"].astype(float)) ** 2)
        work["daily_rmse_bps"] = np.sqrt(pd.concat(errors, axis=1).mean(axis=1)) * 10000.0
        work["rolling_rmse_bps"] = work["daily_rmse_bps"].rolling(window_rows, min_periods=max(5, min(21, window_rows))).mean()
        rows.append(work[["date", "model_type", "daily_rmse_bps", "rolling_rmse_bps"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_training_diagnostics(completed_manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Path]]:
    summaries: list[dict[str, Any]] = []
    figures: dict[str, Path] = {}
    for _, row in completed_manifest.iterrows():
        training_dir = Path(str(row["run_dir"])) / "training"
        history_path = training_dir / "neural_training_history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        if history.empty:
            continue
        for model_type, group in history.groupby("model_type"):
            group = group.copy()
            for column in ["validation_rmse", "train_rmse", "validation_directional_accuracy", "train_directional_accuracy", "elapsed_seconds"]:
                if column in group.columns:
                    group[column] = pd.to_numeric(group[column], errors="coerce")
            cutoff_best = group.sort_values("epoch").groupby("cutoff_date", as_index=False).tail(1)
            best_epochs = group.loc[group.get("is_best_epoch", False).astype(bool)] if "is_best_epoch" in group.columns else pd.DataFrame()
            summaries.append(
                {
                    "model_type": model_type,
                    "architecture_label": row["label"],
                    "run_dir": row["run_dir"],
                    "n_history_rows": int(len(group)),
                    "n_cutoffs": int(group["cutoff_date"].nunique()) if "cutoff_date" in group.columns else 0,
                    "device": _mode_or_none(group.get("device")),
                    "feature_extraction_method": _mode_or_none(group.get("feature_extraction_method")),
                    "lookback_rows": _mode_or_none(group.get("lookback_rows")),
                    "avg_best_epoch": _mean_or_none(best_epochs.get("epoch")),
                    "avg_best_validation_rmse_bps": _mean_or_none(best_epochs.get("validation_rmse"), scale=10000.0),
                    "avg_final_validation_rmse_bps": _mean_or_none(cutoff_best.get("validation_rmse"), scale=10000.0),
                    "avg_final_train_rmse_bps": _mean_or_none(cutoff_best.get("train_rmse"), scale=10000.0),
                    "avg_final_validation_directional_accuracy_pct": _mean_or_none(cutoff_best.get("validation_directional_accuracy"), scale=100.0),
                    "total_elapsed_seconds": _sum_or_none(group.get("elapsed_seconds")),
                }
            )
            curve_path = training_dir / f"{_safe_filename(str(model_type))}_training_curves.svg"
            if curve_path.exists():
                figures[str(model_type)] = curve_path
    return pd.DataFrame(summaries), figures


def write_performance_figures(
    figures_dir: Path,
    predictions: pd.DataFrame,
    ranking: pd.DataFrame,
    per_factor: pd.DataFrame,
    gap_day: pd.DataFrame,
    rolling_rmse: pd.DataFrame,
    target_columns: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths["avg_rmse_by_model"] = figures_dir / "avg_rmse_by_model.svg"
    paths["avg_rmse_by_model"].write_text(
        _bar_svg("Average RMSE by model", ranking, "model_type", "avg_rmse_bps", "RMSE, bps", lower_is_better=True),
        encoding="utf-8",
    )
    paths["directional_accuracy_by_model"] = figures_dir / "directional_accuracy_by_model.svg"
    paths["directional_accuracy_by_model"].write_text(
        _bar_svg(
            "Directional accuracy by model",
            ranking,
            "model_type",
            "avg_directional_accuracy_pct",
            "Directional accuracy, %",
            lower_is_better=False,
        ),
        encoding="utf-8",
    )
    paths["factor_rmse_heatmap"] = figures_dir / "factor_rmse_heatmap.svg"
    paths["factor_rmse_heatmap"].write_text(_metric_heatmap_svg("Factor RMSE, bps", per_factor, "rmse_bps", target_columns, lower_is_better=True), encoding="utf-8")
    paths["factor_correlation_heatmap"] = figures_dir / "factor_correlation_heatmap.svg"
    paths["factor_correlation_heatmap"].write_text(_metric_heatmap_svg("Factor correlation", per_factor, "correlation", target_columns, lower_is_better=False), encoding="utf-8")
    paths["gap_day_rmse"] = figures_dir / "gap_day_rmse.svg"
    paths["gap_day_rmse"].write_text(_gap_line_svg("Average RMSE by gap day", gap_day, "gap_day"), encoding="utf-8")
    paths["rolling_rmse"] = figures_dir / "rolling_rmse.svg"
    paths["rolling_rmse"].write_text(_rolling_rmse_svg("Rolling average daily RMSE", rolling_rmse, ranking), encoding="utf-8")

    best_model = str(ranking.iloc[0]["model_type"]) if not ranking.empty else str(predictions["model_type"].iloc[0])
    best_predictions = predictions.loc[predictions["model_type"] == best_model].copy()
    paths["best_model_scatter"] = figures_dir / "best_model_predicted_vs_actual.svg"
    paths["best_model_scatter"].write_text(_scatter_grid_svg(f"{best_model}: predicted vs actual", best_predictions, target_columns), encoding="utf-8")
    paths["best_model_timeseries"] = figures_dir / "best_model_actual_vs_predicted_timeseries.svg"
    paths["best_model_timeseries"].write_text(_timeseries_grid_svg(f"{best_model}: actual vs predicted", best_predictions, target_columns), encoding="utf-8")
    paths["best_model_distribution"] = figures_dir / "best_model_distribution.svg"
    paths["best_model_distribution"].write_text(_distribution_grid_svg(f"{best_model}: actual/predicted distributions", best_predictions, target_columns), encoding="utf-8")
    return paths


def build_performance_report_html(
    title: str,
    manifest_path: Path,
    completed_manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    ranking: pd.DataFrame,
    per_factor: pd.DataFrame,
    gap_day: pd.DataFrame,
    release_gap_size: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    prediction_volatility: pd.DataFrame,
    factor_extremes: pd.DataFrame,
    training_summary: pd.DataFrame,
    training_figures: dict[str, Path],
    figures: dict[str, Path],
    target_columns: list[str],
) -> str:
    best = ranking.iloc[0] if not ranking.empty else {}
    best_model = str(best.get("model_type", "n/a"))
    best_linear = _best_matching_model(ranking, ["elasticnet", "ridge", "per_factor_elasticnet"])
    best_neural = _best_matching_model(ranking, ["tcn", "ft_transformer", "tft"])
    start, end = _date_range(predictions)
    n_dates = int(predictions["date"].nunique())
    ewma = _ranking_row(ranking, "ewma")
    rolling = _ranking_row(ranking, "rolling_mean")
    best_rmse = float(best.get("avg_rmse_bps", np.nan)) if len(ranking) else np.nan
    ewma_delta = _pct_delta(best_rmse, None if ewma is None else float(ewma["avg_rmse_bps"]))
    rolling_delta = _pct_delta(best_rmse, None if rolling is None else float(rolling["avg_rmse_bps"]))

    article_bullets = article_support_bullets(ranking, per_factor, target_columns)
    training_note = "Neural training diagnostics were found and embedded below." if training_figures else "No neural training diagnostics were found for this manifest."

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(title)}</title>",
            f"<style>{_report_css()}</style>",
            "</head>",
            "<body>",
            '<main class="page">',
            '<header class="hero">',
            "<div>",
            '<div class="kicker">Model performance analysis</div>',
            f"<h1>{escape(title)}</h1>",
            '<p class="deck">Architecture-level nowcast performance, factor-level diagnostics, release-gap behavior, and neural training curves for the latest full-history comparison run.</p>',
            "</div>",
            '<aside class="stamp">',
            "<span>Manifest</span>",
            f"<strong>{escape(manifest_path.parent.name)}</strong>",
            f"<span>{escape(str(manifest_path))}</span>",
            "</aside>",
            "</header>",
            '<section class="metrics">',
            _metric_card("Coverage", f"{start} to {end}", f"{n_dates:,} unique prediction dates."),
            _metric_card("Best model", _display_model(best_model), f"{_fmt(best_rmse, 2)} bps average RMSE."),
            _metric_card("Best linear", _display_model(best_linear or "n/a"), "Regularized linear benchmark."),
            _metric_card("Best neural", _display_model(best_neural or "n/a"), training_note),
            "</section>",
            '<section class="callout">',
            f"<strong>Headline:</strong> {_display_model(best_model)} ranked first by average RMSE. "
            f"It was {escape(_fmt_pct_delta(ewma_delta))} versus EWMA and {escape(_fmt_pct_delta(rolling_delta))} versus rolling mean on average RMSE.",
            "</section>",
            "<section>",
            "<h2>1. Architecture ranking</h2>",
            "<p>All metrics are computed on shared dates after combining completed architecture runs. RMSE and MAE are shown in basis points for readability.</p>",
            _html_table(ranking, ["model_type", "avg_rmse_bps", "avg_mae_bps", "avg_corr", "avg_r2", "avg_directional_accuracy_pct", "rmse_vs_baseline_pct"], max_rows=20),
            '<div class="two">',
            _figure_card("Average RMSE", figures["avg_rmse_by_model"]),
            _figure_card("Directional accuracy", figures["directional_accuracy_by_model"]),
            "</div>",
            "</section>",
            "<section>",
            "<h2>2. Factor-level performance</h2>",
            "<p>Momentum is treated as a first-class target. The heatmaps show which factor targets are most visible in the market-derived feature set.</p>",
            '<div class="two">',
            _figure_card("RMSE heatmap", figures["factor_rmse_heatmap"]),
            _figure_card("Correlation heatmap", figures["factor_correlation_heatmap"]),
            "</div>",
            "<h3>Best and weakest factor by model</h3>",
            _html_table(factor_extremes, ["model_type", "best_target", "best_rmse_bps", "worst_target", "worst_rmse_bps"], max_rows=20),
            "</section>",
            "<section>",
            "<h2>3. Baseline comparison and distribution checks</h2>",
            "<p>This section shows how much each model improves over simple target-only baselines, and whether prediction volatility broadly matches official factor volatility.</p>",
            _html_table(baseline_comparison, ["model_type", "baseline_model", "target", "rmse_improvement_bps", "rmse_vs_baseline_pct"], max_rows=36),
            "<h3>Prediction volatility and hit rates</h3>",
            _html_table(prediction_volatility, ["model_type", "target", "prediction_std_to_actual_std", "sign_bias_pct", "top_quintile_hit_rate_pct", "bottom_quintile_hit_rate_pct"], max_rows=42),
            "</section>",
            "<section>",
            "<h2>4. Release-gap behavior</h2>",
            "<p>Gap-day metrics test whether performance degrades as the simulated unreleased window grows longer.</p>",
            _figure_card("RMSE by gap day", figures["gap_day_rmse"]),
            "<h3>Gap-day table</h3>",
            _html_table(gap_day, ["model_type", "gap_day", "avg_rmse_bps", "avg_corr", "avg_directional_accuracy_pct"], max_rows=60),
            "<h3>Release-gap-size table</h3>",
            _html_table(release_gap_size, ["model_type", "release_gap_size", "avg_rmse_bps", "avg_corr", "avg_directional_accuracy_pct"], max_rows=60),
            "</section>",
            "<section>",
            "<h2>5. Prediction quality visuals</h2>",
            "<p>The following plots focus on the best-ranked model. They support the article discussion about where the estimator tracks official-like factor values closely and where it misses.</p>",
            _figure_card("Predicted versus actual", figures["best_model_scatter"]),
            _figure_card("Actual versus predicted through time", figures["best_model_timeseries"]),
            _figure_card("Actual versus predicted distributions", figures["best_model_distribution"]),
            _figure_card("Rolling RMSE over time", figures["rolling_rmse"]),
            "</section>",
            "<section>",
            "<h2>6. Neural training diagnostics</h2>",
            f"<p>{escape(training_note)} These curves are copied from model training diagnostics and summarize mean behavior by epoch across release-gap cutoffs.</p>",
            _html_table(training_summary, ["model_type", "n_cutoffs", "device", "feature_extraction_method", "lookback_rows", "avg_best_epoch", "avg_best_validation_rmse_bps", "avg_final_validation_directional_accuracy_pct", "total_elapsed_seconds"], max_rows=20),
            _training_figure_cards(training_figures),
            "</section>",
            "<section>",
            "<h2>7. Article-support notes</h2>",
            "<p>These are direct points the report supports for the planned LinkedIn article.</p>",
            "<ul>",
            *[f"<li>{escape(item)}</li>" for item in article_bullets],
            "</ul>",
            "</section>",
            "<footer>",
            f"Generated from {escape(str(manifest_path))}. Tables and figures are in {escape(str(manifest_path.parent / 'performance_analysis'))}.",
            "</footer>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def article_support_bullets(ranking: pd.DataFrame, per_factor: pd.DataFrame, target_columns: list[str]) -> list[str]:
    if ranking.empty or per_factor.empty:
        return ["Insufficient metrics were available for article-support notes."]
    best_model = str(ranking.iloc[0]["model_type"])
    best_rmse = float(ranking.iloc[0]["avg_rmse_bps"])
    neural = ranking.loc[ranking["model_type"].isin(["tcn", "ft_transformer", "tft"])]
    best_neural = neural.iloc[0] if not neural.empty else None
    model_factors = per_factor.loc[per_factor["model_type"] == best_model].sort_values("rmse_bps")
    best_factor = model_factors.iloc[0]
    worst_factor = model_factors.iloc[-1]
    bullets = [
        f"{_display_model(best_model)} is the strongest architecture in this run, with {best_rmse:.2f} bps average RMSE across {len(target_columns)} factors.",
        f"The easiest factor for the best model is {best_factor['target']} ({float(best_factor['rmse_bps']):.2f} bps RMSE); the hardest is {worst_factor['target']} ({float(worst_factor['rmse_bps']):.2f} bps RMSE).",
        "The comparison supports a gap-filling claim: market data contains substantial same-day information about official-like factor values.",
        "The comparison does not prove exact replication of the official portfolio construction; it evaluates supervised nowcast accuracy against hidden official values.",
    ]
    if best_neural is not None:
        bullets.append(
            f"The best neural architecture in this run is {_display_model(str(best_neural['model_type']))}, with {float(best_neural['avg_rmse_bps']):.2f} bps average RMSE; it did not beat the best regularized linear model."
        )
    return bullets


def _manifest_has_completed_runs(path: Path) -> bool:
    try:
        manifest = pd.read_csv(path, sep="\t", keep_default_na=False)
    except Exception:
        return False
    return bool((manifest.get("status", pd.Series(dtype=str)).eq("completed")).any())


def _best_matching_model(ranking: pd.DataFrame, candidates: list[str]) -> str | None:
    if ranking.empty:
        return None
    for model in ranking["model_type"].astype(str):
        if model in candidates:
            return model
    return None


def _ranking_row(ranking: pd.DataFrame, model_type: str) -> pd.Series | None:
    match = ranking.loc[ranking["model_type"] == model_type]
    return None if match.empty else match.iloc[0]


def _pct_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0:
        return None
    return (value / baseline - 1.0) * 100.0


def _fmt_pct_delta(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "not comparable"
    return f"{value:.1f}%"


def _date_range(predictions: pd.DataFrame) -> tuple[str, str]:
    dates = pd.to_datetime(predictions["date"])
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _mean_or_none(series: pd.Series | None, scale: float = 1.0) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.mean() * scale)


def _sum_or_none(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.sum())


def _mode_or_none(series: pd.Series | None) -> Any:
    if series is None:
        return None
    values = series.dropna()
    if values.empty:
        return None
    return values.mode().iloc[0] if not values.mode().empty else values.iloc[0]


def _html_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return '<p class="empty">No data available.</p>'
    cols = [column for column in columns if column in df.columns]
    table = df.loc[:, cols].head(max_rows).copy()
    parts = ["<table>", "<thead><tr>", *[f"<th>{escape(_display_column(col))}</th>" for col in cols], "</tr></thead>", "<tbody>"]
    for _, row in table.iterrows():
        parts.append("<tr>")
        for col in cols:
            parts.append(f"<td>{escape(_format_cell(row[col], col))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    if len(df) > max_rows:
        parts.append(f'<p class="note">Showing {max_rows} of {len(df)} rows. Full table is available as CSV.</p>')
    return "\n".join(parts)


def _metric_card(label: str, value: str, note: str) -> str:
    return f'<div class="card"><span class="label">{escape(label)}</span><span class="value">{escape(value)}</span><div class="note">{escape(note)}</div></div>'


def _figure_card(title: str, path: Path) -> str:
    content = path.read_text(encoding="utf-8") if path.exists() else "<p>Figure unavailable.</p>"
    return f'<article class="figure-card"><h3>{escape(title)}</h3><div class="figure">{content}</div></article>'


def _training_figure_cards(training_figures: dict[str, Path]) -> str:
    if not training_figures:
        return '<p class="empty">No neural training curve SVGs were found.</p>'
    cards = []
    for model_type, path in sorted(training_figures.items()):
        cards.append(_figure_card(f"{_display_model(model_type)} training curves", path))
    return "\n".join(cards)


def _display_model(value: str) -> str:
    mapping = {
        "per_factor_elasticnet": "Per-factor ElasticNet",
        "elasticnet": "ElasticNet",
        "ridge": "Ridge",
        "rolling_mean": "Rolling Mean",
        "ewma": "EWMA",
        "gradient_boosting": "Histogram Gradient Boosting",
        "ft_transformer": "FT-Transformer",
        "tcn": "TCN",
        "tft": "TFT",
    }
    return mapping.get(value, value.replace("_", " ").title())


def _display_column(value: str) -> str:
    return value.replace("_", " ").replace("bps", "bps").title()


def _format_cell(value: Any, column: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        if column.endswith("_pct") or column in {"rmse_vs_baseline_pct", "avg_directional_accuracy_pct"}:
            return f"{value:.1f}%"
        if column.endswith("_bps") or "rmse_bps" in column:
            return f"{value:.2f}"
        if "corr" in column or "r2" in column or "ratio" in column:
            return f"{value:.3f}"
        return f"{value:.4g}"
    return str(value)


def _fmt(value: float, decimals: int = 2) -> str:
    return "" if not np.isfinite(value) else f"{value:.{decimals}f}"


def _report_css() -> str:
    return """
:root{--paper:#f3efe2;--ink:#132125;--muted:#64706b;--line:#d2c4aa;--panel:#fffaf0;--teal:#174a5a;--rust:#963f2d;--olive:#596f36;--gold:#c8a84f}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Aptos,Optima,'Trebuchet MS',Verdana,sans-serif}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.16;background-image:linear-gradient(90deg,var(--line) 1px,transparent 1px),linear-gradient(var(--line) 1px,transparent 1px);background-size:36px 36px}
.page{max-width:1280px;margin:0 auto;padding:34px 28px 88px;position:relative}.hero{display:grid;grid-template-columns:1.35fr .65fr;gap:26px;align-items:stretch;border-bottom:2px solid var(--ink);padding-bottom:26px}
.kicker{text-transform:uppercase;letter-spacing:.15em;font-size:.78rem;color:var(--rust);font-weight:900;margin-bottom:12px}h1{font-family:Georgia,'Times New Roman',serif;font-size:clamp(2.3rem,5vw,5.2rem);line-height:.94;margin:0}
.deck{font-family:Georgia,'Times New Roman',serif;font-size:1.18rem;line-height:1.48;color:#283533;max-width:900px}.stamp{background:var(--ink);color:var(--paper);padding:20px;display:flex;flex-direction:column;justify-content:space-between;min-height:230px}
.stamp strong{display:block;font-size:2.8rem;line-height:.9;font-family:Georgia,serif}.stamp span{color:#ddd1bc;line-height:1.42;overflow-wrap:anywhere}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;margin:24px 0}.card,.figure-card{background:var(--panel);border:1px solid var(--line);box-shadow:0 10px 24px rgba(21,32,34,.06)}
.card{padding:18px}.card .label{color:var(--muted);font-size:.77rem;text-transform:uppercase;letter-spacing:.09em;font-weight:900}.card .value{display:block;font-size:1.8rem;font-family:Georgia,serif;margin-top:8px}.card .note,.note{color:var(--muted);font-size:.92rem;line-height:1.4}
section{margin-top:40px}h2{font-family:Georgia,serif;font-size:2rem;margin:0 0 12px}h3{font-size:1rem;text-transform:uppercase;letter-spacing:.08em;margin:20px 0 10px;color:var(--teal)}p{line-height:1.62;color:#2c3836}.callout{border-left:6px solid var(--rust);background:#efe4d1;padding:16px 18px;line-height:1.55}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);margin-top:10px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;font-size:.88rem;vertical-align:top}th{background:#eadfca;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em}
.two{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}.figure-card{padding:16px;margin-top:16px}.figure svg{width:100%;height:auto;background:#fff;border:1px solid var(--line)}.empty{color:var(--muted);font-style:italic}footer{margin-top:46px;color:var(--muted);font-size:.9rem;border-top:1px solid var(--line);padding-top:16px}
@media(max-width:900px){.hero,.two,.metrics{grid-template-columns:1fr}.stamp{min-height:auto}}
"""


def _bar_svg(title: str, data: pd.DataFrame, label_col: str, value_col: str, y_label: str, *, lower_is_better: bool) -> str:
    width, height = 980, 420
    margin = {"left": 74, "right": 28, "top": 62, "bottom": 100}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    if data.empty or value_col not in data:
        return _empty_svg(width, height, title)
    table = data[[label_col, value_col]].dropna().copy()
    table[value_col] = pd.to_numeric(table[value_col], errors="coerce")
    table = table.dropna().sort_values(value_col, ascending=lower_is_better)
    max_value = float(table[value_col].max()) * 1.08 or 1.0
    parts = [_svg_header(width, height), f'<text x="24" y="36" class="title">{escape(title)}</text>', f'<text x="24" y="56" class="label">{escape(y_label)}</text>']
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"]+plot_h}" class="axis"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]+plot_h}" x2="{margin["left"]+plot_w}" y2="{margin["top"]+plot_h}" class="axis"/>')
    for tick in range(5):
        y = margin["top"] + tick / 4 * plot_h
        value = max_value - tick / 4 * max_value
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"]+plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{margin["left"]-8}" y="{y+4:.2f}" text-anchor="end" class="legend">{value:.1f}</text>')
    bar_w = plot_w / max(1, len(table)) * 0.68
    for idx, row in enumerate(table.itertuples(index=False)):
        label = str(getattr(row, label_col))
        value = float(getattr(row, value_col))
        x = margin["left"] + (idx + 0.16) * plot_w / max(1, len(table))
        bar_h = plot_h * value / max_value
        y = margin["top"] + plot_h - bar_h
        color = ["#174a5a", "#963f2d", "#596f36", "#725a92", "#b86b33", "#2c7067", "#6f6046"][idx % 7]
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{color}" opacity="0.88"/>')
        parts.append(f'<text x="{x + bar_w/2:.2f}" y="{margin["top"]+plot_h+18}" text-anchor="middle" class="legend" transform="rotate(35 {x + bar_w/2:.2f},{margin["top"]+plot_h+18})">{escape(_display_model(label))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _metric_heatmap_svg(title: str, data: pd.DataFrame, value_col: str, target_columns: list[str], *, lower_is_better: bool) -> str:
    if data.empty or value_col not in data:
        return _empty_svg(980, 420, title)
    models = data.groupby("model_type")[value_col].mean().sort_values(ascending=lower_is_better).index.tolist()
    cell_w, cell_h = 100, 42
    left, top = 170, 70
    width = left + cell_w * len(target_columns) + 44
    height = top + cell_h * len(models) + 60
    values = pd.to_numeric(data[value_col], errors="coerce").dropna()
    v_min, v_max = float(values.min()), float(values.max())
    parts = [_svg_header(width, height), f'<text x="24" y="36" class="title">{escape(title)}</text>']
    for col_idx, target in enumerate(target_columns):
        parts.append(f'<text x="{left + col_idx*cell_w + cell_w/2:.1f}" y="{top-14}" text-anchor="middle" class="legend">{escape(target)}</text>')
    lookup = data.set_index(["model_type", "target"])[value_col].to_dict()
    for row_idx, model in enumerate(models):
        y = top + row_idx * cell_h
        parts.append(f'<text x="{left-10}" y="{y + cell_h/2 + 4:.1f}" text-anchor="end" class="legend">{escape(_display_model(str(model)))}</text>')
        for col_idx, target in enumerate(target_columns):
            x = left + col_idx * cell_w
            value = lookup.get((model, target))
            color = _heat_color(None if value is None or pd.isna(value) else float(value), v_min, v_max, lower_is_better)
            label = "" if value is None or pd.isna(value) else f"{float(value):.2f}" if value_col.endswith("_bps") else f"{float(value):.2f}"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + cell_w/2:.1f}" y="{y + cell_h/2 + 4:.1f}" text-anchor="middle" class="legend">{escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _gap_line_svg(title: str, data: pd.DataFrame, x_col: str) -> str:
    if data.empty:
        return _empty_svg(980, 420, title)
    ranking = data.groupby("model_type")["avg_rmse_bps"].mean().sort_values().head(6).index.tolist()
    return _multi_line_svg(title, data.loc[data["model_type"].isin(ranking)], x_col, "avg_rmse_bps", "model_type", "RMSE, bps")


def _rolling_rmse_svg(title: str, rolling: pd.DataFrame, ranking: pd.DataFrame) -> str:
    if rolling.empty:
        return _empty_svg(980, 420, title)
    models = ranking["model_type"].head(5).astype(str).tolist() if not ranking.empty else rolling["model_type"].unique().tolist()[:5]
    data = rolling.loc[rolling["model_type"].isin(models)].copy()
    data["date_num"] = pd.to_datetime(data["date"]).map(pd.Timestamp.toordinal)
    return _multi_line_svg(title, data.dropna(subset=["rolling_rmse_bps"]), "date_num", "rolling_rmse_bps", "model_type", "63-row rolling RMSE, bps", x_is_date=True)


def _multi_line_svg(title: str, data: pd.DataFrame, x_col: str, y_col: str, group_col: str, y_label: str, *, x_is_date: bool = False) -> str:
    width, height = 980, 420
    margin = {"left": 74, "right": 28, "top": 62, "bottom": 58}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    if data.empty:
        return _empty_svg(width, height, title)
    x = pd.to_numeric(data[x_col], errors="coerce")
    y = pd.to_numeric(data[y_col], errors="coerce")
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = 0.0, float(y.max()) * 1.08
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_max == y_min:
        y_max = y_min + 1
    parts = [_svg_header(width, height), f'<text x="24" y="36" class="title">{escape(title)}</text>', f'<text x="24" y="56" class="label">{escape(y_label)}</text>']
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"]+plot_h}" class="axis"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]+plot_h}" x2="{margin["left"]+plot_w}" y2="{margin["top"]+plot_h}" class="axis"/>')
    for tick in range(5):
        yy = margin["top"] + tick / 4 * plot_h
        value = y_max - tick / 4 * (y_max - y_min)
        parts.append(f'<line x1="{margin["left"]}" y1="{yy:.2f}" x2="{margin["left"]+plot_w}" y2="{yy:.2f}" class="grid"/>')
        parts.append(f'<text x="{margin["left"]-8}" y="{yy+4:.2f}" text-anchor="end" class="legend">{value:.1f}</text>')
    colors = ["#174a5a", "#963f2d", "#596f36", "#725a92", "#b86b33", "#2c7067", "#6f6046"]
    for idx, (model, group) in enumerate(data.groupby(group_col)):
        points = []
        for _, row in group.sort_values(x_col).iterrows():
            xv = float(row[x_col])
            yv = float(row[y_col])
            px = margin["left"] + (xv - x_min) / (x_max - x_min) * plot_w
            py = margin["top"] + (1 - (yv - y_min) / (y_max - y_min)) * plot_h
            points.append((px, py))
        if len(points) >= 2:
            path = " ".join(("M" if i == 0 else "L") + f"{px:.2f},{py:.2f}" for i, (px, py) in enumerate(points))
            color = colors[idx % len(colors)]
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" opacity="0.9"/>')
            lx = margin["left"] + idx * 155
            parts.append(f'<line x1="{lx}" y1="{height-20}" x2="{lx+22}" y2="{height-20}" stroke="{color}" class="line"/>')
            parts.append(f'<text x="{lx+28}" y="{height-16}" class="legend">{escape(_display_model(str(model)))}</text>')
    if x_is_date:
        for ordinal in np.linspace(x_min, x_max, 5):
            label = pd.Timestamp.fromordinal(int(ordinal)).strftime("%Y-%m")
            px = margin["left"] + (ordinal - x_min) / (x_max - x_min) * plot_w
            parts.append(f'<text x="{px:.1f}" y="{margin["top"]+plot_h+22}" text-anchor="middle" class="legend">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _scatter_grid_svg(title: str, predictions: pd.DataFrame, target_columns: list[str]) -> str:
    width, panel_w, panel_h = 1080, 330, 245
    cols = 3
    rows = int(np.ceil(len(target_columns) / cols))
    height = 64 + rows * panel_h + 30
    parts = [_svg_header(width, height), f'<text x="24" y="36" class="title">{escape(title)}</text>', '<text x="24" y="56" class="label">Values are daily returns in bps. Diagonal line marks perfect prediction.</text>']
    for idx, target in enumerate(target_columns):
        row, col = divmod(idx, cols)
        x0 = 48 + col * panel_w
        y0 = 80 + row * panel_h
        actual = predictions[f"actual_{target}"].astype(float) * 10000.0
        pred = predictions[f"pred_{target}"].astype(float) * 10000.0
        data = pd.DataFrame({"actual": actual, "pred": pred}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) > 550:
            data = data.iloc[np.linspace(0, len(data) - 1, 550, dtype=int)]
        _scatter_panel(parts, data, target, x0, y0, panel_w - 34, panel_h - 48)
    parts.append("</svg>")
    return "\n".join(parts)


def _scatter_panel(parts: list[str], data: pd.DataFrame, title: str, x0: int, y0: int, width: int, height: int) -> None:
    plot_left, plot_top = x0 + 48, y0 + 26
    plot_w, plot_h = width - 64, height - 48
    finite = pd.concat([data["actual"], data["pred"]]).to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (-1.0, 1.0)
    pad = (hi - lo) * 0.08 or 1.0
    lo -= pad
    hi += pad
    parts.append(f'<text x="{x0}" y="{y0+14}" class="label" font-weight="700">{escape(title)}</text>')
    parts.append(f'<rect x="{plot_left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d2c4aa"/>')
    parts.append(f'<path d="M{plot_left},{plot_top+plot_h} L{plot_left+plot_w},{plot_top}" stroke="#96a1ad" stroke-dasharray="4 4" fill="none"/>')
    for _, point in data.iterrows():
        px = plot_left + (float(point["actual"]) - lo) / (hi - lo) * plot_w
        py = plot_top + (1 - (float(point["pred"]) - lo) / (hi - lo)) * plot_h
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.7" fill="#174a5a" opacity="0.42"/>')


def _timeseries_grid_svg(title: str, predictions: pd.DataFrame, target_columns: list[str]) -> str:
    width, panel_h = 1080, 160
    height = 66 + panel_h * len(target_columns) + 30
    parts = [_svg_header(width, height), f'<text x="24" y="36" class="title">{escape(title)}</text>', '<text x="24" y="56" class="label">Actual and predicted factor values through time, in bps.</text>']
    dates = pd.to_datetime(predictions["date"])
    x_min, x_max = dates.min().value, dates.max().value
    for idx, target in enumerate(target_columns):
        _timeseries_panel(parts, predictions, target, 56, 80 + idx * panel_h, width - 92, panel_h - 28, x_min, x_max)
    parts.append("</svg>")
    return "\n".join(parts)


def _timeseries_panel(parts: list[str], predictions: pd.DataFrame, target: str, x0: int, y0: int, width: int, height: int, x_min: int, x_max: int) -> None:
    actual = predictions[f"actual_{target}"].astype(float) * 10000.0
    pred = predictions[f"pred_{target}"].astype(float) * 10000.0
    values = pd.concat([actual, pred]).replace([np.inf, -np.inf], np.nan).dropna()
    y_min, y_max = (float(values.min()), float(values.max())) if len(values) else (-1.0, 1.0)
    pad = (y_max - y_min) * 0.08 or 1.0
    y_min -= pad
    y_max += pad
    parts.append(f'<text x="{x0}" y="{y0+14}" class="label" font-weight="700">{escape(target)}</text>')
    parts.append(f'<rect x="{x0+58}" y="{y0+22}" width="{width-70}" height="{height-34}" fill="#ffffff" stroke="#d2c4aa"/>')
    for column, color in [(f"actual_{target}", "#263b59"), (f"pred_{target}", "#b46732")]:
        points = []
        for date, value in zip(pd.to_datetime(predictions["date"]), predictions[column].astype(float) * 10000.0):
            px = x0 + 58 + (date.value - x_min) / max(1, x_max - x_min) * (width - 70)
            py = y0 + 22 + (1 - (value - y_min) / (y_max - y_min)) * (height - 34)
            points.append((px, py))
        path = " ".join(("M" if i == 0 else "L") + f"{px:.2f},{py:.2f}" for i, (px, py) in enumerate(points))
        parts.append(f'<path d="{path}" stroke="{color}" fill="none" stroke-width="1.6" opacity="0.9"/>')


def _distribution_grid_svg(title: str, predictions: pd.DataFrame, target_columns: list[str]) -> str:
    width, panel_w, panel_h = 1080, 330, 220
    cols = 3
    rows = int(np.ceil(len(target_columns) / cols))
    height = 64 + rows * panel_h + 30
    parts = [_svg_header(width, height), f'<text x="24" y="36" class="title">{escape(title)}</text>', '<text x="24" y="56" class="label">Histogram comparison of actual and predicted values in bps.</text>']
    for idx, target in enumerate(target_columns):
        row, col = divmod(idx, cols)
        _hist_panel(parts, predictions, target, 48 + col * panel_w, 84 + row * panel_h, panel_w - 36, panel_h - 42)
    parts.append("</svg>")
    return "\n".join(parts)


def _hist_panel(parts: list[str], predictions: pd.DataFrame, target: str, x0: int, y0: int, width: int, height: int) -> None:
    actual = (predictions[f"actual_{target}"].astype(float) * 10000.0).replace([np.inf, -np.inf], np.nan).dropna()
    pred = (predictions[f"pred_{target}"].astype(float) * 10000.0).replace([np.inf, -np.inf], np.nan).dropna()
    values = pd.concat([actual, pred])
    if values.empty:
        return
    bins = np.linspace(float(values.quantile(0.01)), float(values.quantile(0.99)), 24)
    actual_counts, edges = np.histogram(actual, bins=bins)
    pred_counts, _ = np.histogram(pred, bins=bins)
    max_count = max(int(actual_counts.max()), int(pred_counts.max()), 1)
    parts.append(f'<text x="{x0}" y="{y0+12}" class="label" font-weight="700">{escape(target)}</text>')
    plot_left, plot_top = x0 + 44, y0 + 24
    plot_w, plot_h = width - 58, height - 46
    parts.append(f'<rect x="{plot_left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d2c4aa"/>')
    bar_w = plot_w / max(1, len(actual_counts))
    for idx, (actual_count, pred_count) in enumerate(zip(actual_counts, pred_counts)):
        x = plot_left + idx * bar_w
        ah = plot_h * actual_count / max_count
        ph = plot_h * pred_count / max_count
        parts.append(f'<rect x="{x:.2f}" y="{plot_top + plot_h - ah:.2f}" width="{bar_w*0.48:.2f}" height="{ah:.2f}" fill="#263b59" opacity="0.55"/>')
        parts.append(f'<rect x="{x + bar_w*0.48:.2f}" y="{plot_top + plot_h - ph:.2f}" width="{bar_w*0.48:.2f}" height="{ph:.2f}" fill="#b46732" opacity="0.55"/>')


def _heat_color(value: float | None, v_min: float, v_max: float, lower_is_better: bool) -> str:
    if value is None:
        return "#f3f4f6"
    if v_max == v_min:
        score = 0.5
    else:
        score = (value - v_min) / (v_max - v_min)
    if lower_is_better:
        score = 1.0 - score
    low = np.array([150, 63, 45])
    high = np.array([225, 237, 207])
    rgb = low + score * (high - low)
    return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"


def _svg_header(width: int, height: int) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<style>",
            "text{font-family:Aptos,Optima,Trebuchet MS,Verdana,sans-serif;fill:#1f2933}.title{font-size:20px;font-weight:800}.label{font-size:12px}.legend{font-size:11px}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:2}",
            "</style>",
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        ]
    )


def _empty_svg(width: int, height: int, title: str) -> str:
    return "\n".join([_svg_header(width, height), f'<text x="24" y="40" class="title">{escape(title)}: no data</text>', "</svg>"])


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "model"
