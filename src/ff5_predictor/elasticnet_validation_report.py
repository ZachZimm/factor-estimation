from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
import json
import webbrowser

import numpy as np
import pandas as pd

from ff5_predictor.io import ensure_dir
from ff5_predictor.nowcast_io import write_json


DEFAULT_VALIDATION_ROOT = Path("data/nowcasts/elasticnet_time_series_validation_v1")
TARGET_ORDER = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


@dataclass(frozen=True)
class ElasticNetValidationReportResult:
    run_dir: Path
    output_dir: Path
    html_path: Path
    metadata: dict[str, Any]


def run_elasticnet_validation_report(
    run_dir: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    title: str | None = None,
    open_browser: bool = False,
) -> ElasticNetValidationReportResult:
    resolved_run_dir = find_latest_validation_run() if run_dir is None else Path(run_dir)
    if not resolved_run_dir.is_dir():
        raise FileNotFoundError(f"ElasticNet validation run directory not found: {resolved_run_dir}")
    output = ensure_dir(Path(output_dir) if output_dir is not None else resolved_run_dir / "analysis" / "fold_validation_report")
    tables_dir = ensure_dir(output / "tables")

    loaded = _load_validation_tables(resolved_run_dir)
    target_columns = _target_columns(loaded["model_summary"])
    summary_cards = _summary_cards(resolved_run_dir, loaded)
    model_summary = _format_model_summary(loaded["model_summary"])
    ex_mom_summary = _ex_momentum_summary(loaded["model_summary"])
    target_rmse = _target_rmse_table(loaded["model_summary"])
    fold_rmse = _fold_rmse_table(loaded["fold_metrics"])
    coefficient_summary = _coefficient_stability_summary(loaded["coefficient_stability"])
    coefficient_by_target = _coefficient_stability_by_target(loaded["coefficient_stability"])
    vintage_summary = _vintage_summary(loaded["vintage_stability"])
    vintage_target_rmse = _freshest_vintage_target_rmse(loaded["vintage_stability"])
    top_features = _top_feature_table(loaded["coefficient_table"], top_n=8)

    model_summary.to_csv(tables_dir / "model_summary_display.csv", index=False)
    ex_mom_summary.to_csv(tables_dir / "model_summary_ex_momentum.csv", index=False)
    target_rmse.to_csv(tables_dir / "target_rmse_bps.csv")
    fold_rmse.to_csv(tables_dir / "fold_rmse_bps.csv")
    coefficient_summary.to_csv(tables_dir / "coefficient_stability_summary.csv", index=False)
    coefficient_by_target.to_csv(tables_dir / "coefficient_stability_by_target.csv", index=False)
    vintage_summary.to_csv(tables_dir / "vintage_stability_summary.csv")
    vintage_target_rmse.to_csv(tables_dir / "freshest_vintage_target_rmse_bps.csv")
    top_features.to_csv(tables_dir / "top_features.csv", index=False)

    report_title = title or "ElasticNet Time-Series Validation Report"
    html = build_elasticnet_validation_report_html(
        report_title,
        resolved_run_dir,
        loaded,
        summary_cards,
        model_summary,
        ex_mom_summary,
        target_rmse,
        fold_rmse,
        coefficient_summary,
        coefficient_by_target,
        vintage_summary,
        vintage_target_rmse,
        top_features,
        target_columns,
    )
    html_path = output / "elasticnet_time_series_validation_report.html"
    html_path.write_text(html, encoding="utf-8")

    metadata = {
        "run_dir": str(resolved_run_dir),
        "output_dir": str(output),
        "html_path": str(html_path),
        "target_columns": target_columns,
        "table_paths": {
            "model_summary_display": str(tables_dir / "model_summary_display.csv"),
            "model_summary_ex_momentum": str(tables_dir / "model_summary_ex_momentum.csv"),
            "target_rmse_bps": str(tables_dir / "target_rmse_bps.csv"),
            "fold_rmse_bps": str(tables_dir / "fold_rmse_bps.csv"),
            "coefficient_stability_summary": str(tables_dir / "coefficient_stability_summary.csv"),
            "coefficient_stability_by_target": str(tables_dir / "coefficient_stability_by_target.csv"),
            "vintage_stability_summary": str(tables_dir / "vintage_stability_summary.csv"),
            "freshest_vintage_target_rmse_bps": str(tables_dir / "freshest_vintage_target_rmse_bps.csv"),
            "top_features": str(tables_dir / "top_features.csv"),
        },
    }
    write_json(output / "metadata.json", metadata)
    if open_browser:
        webbrowser.open(html_path.resolve().as_uri())
    return ElasticNetValidationReportResult(resolved_run_dir, output, html_path, metadata)


def find_latest_validation_run(root: str | Path = DEFAULT_VALIDATION_ROOT) -> Path:
    base = Path(root)
    candidates = [
        path
        for path in sorted(base.glob("*"))
        if path.is_dir() and (path / "tables" / "model_summary.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed ElasticNet validation runs found under {base}")
    return candidates[-1]


def build_elasticnet_validation_report_html(
    title: str,
    run_dir: Path,
    loaded: dict[str, Any],
    summary_cards: list[dict[str, str]],
    model_summary: pd.DataFrame,
    ex_mom_summary: pd.DataFrame,
    target_rmse: pd.DataFrame,
    fold_rmse: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    coefficient_by_target: pd.DataFrame,
    vintage_summary: pd.DataFrame,
    vintage_target_rmse: pd.DataFrame,
    top_features: pd.DataFrame,
    target_columns: list[str],
) -> str:
    figure_html = _embedded_figures(run_dir)
    card_html = "\n".join(
        f"""<article class="metric-card">
          <span>{escape(card["label"])}</span>
          <strong>{escape(card["value"])}</strong>
          <small>{escape(card["detail"])}</small>
        </article>"""
        for card in summary_cards
    )
    target_note = ", ".join(target_columns)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(title)}</title>
  <style>
    :root {{
      --paper: #f8f2e8;
      --ink: #17242c;
      --muted: #6c746f;
      --line: #d7c8b1;
      --panel: #fffaf2;
      --blue: #164d5f;
      --copper: #b36835;
      --green: #677b42;
      --shadow: rgba(32, 28, 24, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.45;
    }}
    .page {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 42px 24px 72px;
    }}
    header {{
      border-top: 8px solid var(--blue);
      border-bottom: 1px solid var(--line);
      padding: 28px 0 30px;
      margin-bottom: 28px;
    }}
    h1, h2, h3 {{
      font-family: "Trebuchet MS", Verdana, sans-serif;
      letter-spacing: 0;
      line-height: 1.05;
    }}
    h1 {{
      font-size: clamp(2.4rem, 5vw, 5.8rem);
      max-width: 980px;
      margin: 0 0 18px;
      color: var(--blue);
    }}
    h2 {{
      font-size: clamp(1.45rem, 2.2vw, 2.15rem);
      color: var(--blue);
      margin: 42px 0 12px;
    }}
    h3 {{
      color: var(--copper);
      margin: 26px 0 8px;
    }}
    p {{
      max-width: 900px;
      font-size: 1.05rem;
    }}
    .deck {{
      font-size: 1.24rem;
      color: var(--muted);
      max-width: 1040px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 22px 0 28px;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: 0 10px 28px var(--shadow);
      padding: 18px 18px 16px;
      min-height: 128px;
    }}
    .metric-card span {{
      display: block;
      color: var(--muted);
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric-card strong {{
      display: block;
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 1.75rem;
      color: var(--ink);
      margin: 10px 0 8px;
    }}
    .metric-card small {{ color: var(--muted); }}
    .note {{
      border-left: 5px solid var(--copper);
      background: rgba(255, 250, 242, 0.78);
      padding: 14px 18px;
      margin: 22px 0;
      max-width: 1000px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      margin: 14px 0 28px;
      background: var(--panel);
      box-shadow: 0 10px 28px var(--shadow);
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 9px;
      vertical-align: top;
    }}
    th {{
      background: #efe1cb;
      color: var(--blue);
      font-family: "Trebuchet MS", Verdana, sans-serif;
      text-align: left;
      position: sticky;
      top: 0;
    }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .table-scroll {{ overflow-x: auto; }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    .figure-box {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 12px;
      box-shadow: 0 10px 28px var(--shadow);
      overflow-x: auto;
    }}
    .figure-box svg {{ max-width: 100%; height: auto; display: block; }}
    .callouts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin: 18px 0 30px;
    }}
    .callout {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      box-shadow: 0 10px 28px var(--shadow);
    }}
    .callout h3 {{ margin-top: 0; }}
    code {{
      background: #efe1cb;
      padding: 2px 5px;
      border-radius: 3px;
    }}
  </style>
</head>
<body>
  <main class="page">
    <header>
      <h1>{escape(title)}</h1>
      <p class="deck">A fold-based readout of fixed-hyperparameter shared ElasticNet and per-factor ElasticNet, focused on out-of-sample accuracy, coefficient stability, and model-vintage drift.</p>
    </header>

    <section class="cards">
      {card_html}
    </section>

    <section class="note">
      <strong>Interpretive headline:</strong> the fixed ElasticNet models are accurate enough to support a nowcasting discussion, but the coefficient path is not perfectly invariant. The data supports a careful claim: stale model vintages remain usable, while retraining still changes the implied measurement enough that model vintage should be documented.
    </section>

    <h2>1. What Was Validated</h2>
    <p>The validation uses market-only inputs and official Fama-French factors as labels. The target set is {escape(target_note)}. Target lags, recursive factor lags, feature extraction, PCA, PLS, and hyperparameter search are disabled for this study.</p>
    <p>Each fold trains strictly before its validation block. Sliding folds use the latest fixed-size historical window; expanding folds use all available history before the validation block. The separate vintage holdout asks how predictions and coefficients change when the same final holdout window is predicted by increasingly stale model vintages.</p>

    <h2>2. Overall Accuracy</h2>
    <p>Average RMSE is close across the four protocol/model combinations. Sliding-window shared ElasticNet is the best on average by a small margin, while per-factor ElasticNet mainly improves the market factor and does not dominate the full target set.</p>
    {_html_table(model_summary, max_rows=40)}

    <h3>Excluding Momentum</h3>
    <p>Momentum is the hardest target in this setup. Removing it from the average makes the non-momentum factors look substantially tighter.</p>
    {_html_table(ex_mom_summary, max_rows=20)}

    <h3>Target-Level RMSE, Basis Points</h3>
    <p>The model is strongest for <code>Mkt-RF</code>, <code>SMB</code>, and <code>HML</code>. <code>RMW</code>, <code>CMA</code>, and especially <code>Mom</code> remain the weak points.</p>
    {_html_table(target_rmse.reset_index(), max_rows=20)}

    <h2>3. Fold Behavior Over Time</h2>
    <p>The hardest validation year is the 2022 block, while 2023-2025 blocks are more benign. This is a useful article point: average performance is strong, but yearly error is regime-dependent.</p>
    {_html_table(fold_rmse.reset_index(), max_rows=12)}

    <h2>4. Coefficient Stability</h2>
    <p>Expanding-window coefficients are more stable than sliding-window coefficients. Per-factor ElasticNet selects fewer nonzero features and has stronger sign agreement, but its top-feature overlap is not always better.</p>
    {_html_table(coefficient_summary, max_rows=20)}

    <h3>Coefficient Stability by Factor</h3>
    <p><code>Mkt-RF</code>, <code>SMB</code>, and <code>HML</code> have the clearest coefficient stability. <code>CMA</code> and <code>Mom</code> are more fragile, which matches their weaker prediction accuracy.</p>
    {_html_table(coefficient_by_target, max_rows=40)}

    <h2>5. Vintage/Staleness Test</h2>
    <p>The vintage test predicts the same final 252-row holdout using model vintages from 0 to 1,260 rows stale. RMSE deteriorates gradually rather than collapsing. Prediction drift rises more clearly than error, and coefficient correlation drops as stale vintages get older.</p>
    {_html_table(vintage_summary.reset_index(), max_rows=40)}

    <h3>Freshest-Vintage Holdout RMSE, Basis Points</h3>
    {_html_table(vintage_target_rmse.reset_index(), max_rows=20)}

    <h2>6. Feature Interpretation</h2>
    <p>The top coefficients reinforce that the model is not discovering a mysterious black-box signal. The market factor is driven by same-day broad equity ETF returns. Momentum leans on explicit momentum/breadth and value-growth spread proxies.</p>
    {_html_table(top_features, max_rows=64)}

    <h2>7. Figures</h2>
    <div class="figure-grid">
      {figure_html}
    </div>

    <h2>8. How To Use This In The Article</h2>
    <div class="callouts">
      <article class="callout">
        <h3>Accuracy Claim</h3>
        <p>The model produces low daily errors for most factors, especially market, size, and value. Momentum is useful but materially noisier.</p>
      </article>
      <article class="callout">
        <h3>Internal Consistency Claim</h3>
        <p>The coefficients are stable enough to support the idea of a coherent model-implied measurement, but not so stable that retraining vintage is irrelevant.</p>
      </article>
      <article class="callout">
        <h3>Stale Data Claim</h3>
        <p>Using a stale model degrades gradually. That supports practical gap filling when official data is stale, while still arguing for periodic retraining.</p>
      </article>
    </div>
  </main>
</body>
</html>"""


def _load_validation_tables(run_dir: Path) -> dict[str, Any]:
    tables = run_dir / "tables"
    predictions = run_dir / "predictions"
    metadata_path = run_dir / "metadata" / "validation_metadata.json"
    required = {
        "model_summary": tables / "model_summary.csv",
        "fold_metrics": tables / "fold_metrics.csv",
        "coefficient_table": tables / "coefficient_table.csv",
        "coefficient_stability": tables / "coefficient_stability.csv",
        "vintage_stability": tables / "vintage_stability.csv",
        "fold_predictions": predictions / "fold_predictions.csv",
        "vintage_predictions": predictions / "vintage_holdout_predictions.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Validation run is missing required files: {missing}")
    loaded = {name: pd.read_csv(path) for name, path in required.items()}
    loaded["metadata"] = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return loaded


def _target_columns(model_summary: pd.DataFrame) -> list[str]:
    targets = [target for target in model_summary["target"].dropna().astype(str).unique() if target != "__average__"]
    ordered = [target for target in TARGET_ORDER if target in targets]
    return ordered + sorted(set(targets).difference(ordered))


def _summary_cards(run_dir: Path, loaded: dict[str, Any]) -> list[dict[str, str]]:
    metadata = loaded.get("metadata", {})
    model_summary = loaded["model_summary"]
    average = model_summary.loc[model_summary["target"].eq("__average__")].copy()
    best = average.sort_values("avg_rmse").iloc[0] if not average.empty else None
    vintage = loaded["vintage_stability"].copy()
    stale_rows = sorted(vintage["staleness_rows"].dropna().astype(int).unique()) if not vintage.empty else []
    return [
        {
            "label": "Run",
            "value": run_dir.name,
            "detail": f"{metadata.get('date_start', '?')} to {metadata.get('date_end', '?')}",
        },
        {
            "label": "Feature Count",
            "value": f"{int(metadata.get('n_features', 0)):,}",
            "detail": "Market-only features, no factor lags",
        },
        {
            "label": "Validation Folds",
            "value": str(int(metadata.get("n_folds", 0))),
            "detail": "Sliding plus expanding annual blocks",
        },
        {
            "label": "Best Average RMSE",
            "value": f"{float(best['avg_rmse']) * 10000:.2f} bps" if best is not None else "n/a",
            "detail": f"{best['protocol']} / {best['model_type']}" if best is not None else "No summary rows",
        },
        {
            "label": "Vintage Span",
            "value": f"{max(stale_rows):,} rows" if stale_rows else "n/a",
            "detail": "Maximum model staleness tested",
        },
    ]


def _format_model_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    average = model_summary.loc[model_summary["target"].eq("__average__")].copy()
    result = pd.DataFrame(
        {
            "protocol": average["protocol"],
            "model_type": average["model_type"],
            "n_folds": average["n_folds"].astype(int),
            "avg_rmse_bps": average["avg_rmse"] * 10000.0,
            "avg_mae_bps": average["avg_mae"] * 10000.0,
            "avg_corr": average["avg_corr"],
            "directional_accuracy_pct": average["avg_directional_accuracy"] * 100.0,
        }
    )
    return result.sort_values("avg_rmse_bps").reset_index(drop=True).round(4)


def _ex_momentum_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    non_mom = model_summary.loc[~model_summary["target"].isin(["__average__", "Mom"])].copy()
    rows = []
    for (protocol, model_type), group in non_mom.groupby(["protocol", "model_type"]):
        rows.append(
            {
                "protocol": protocol,
                "model_type": model_type,
                "avg_rmse_ex_mom_bps": float(group["avg_rmse"].mean() * 10000.0),
                "avg_mae_ex_mom_bps": float(group["avg_mae"].mean() * 10000.0),
                "avg_corr_ex_mom": float(group["avg_corr"].mean()),
                "directional_accuracy_ex_mom_pct": float(group["avg_directional_accuracy"].mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_rmse_ex_mom_bps").reset_index(drop=True).round(4)


def _target_rmse_table(model_summary: pd.DataFrame) -> pd.DataFrame:
    target_rows = model_summary.loc[~model_summary["target"].eq("__average__")].copy()
    target_rows["rmse_bps"] = target_rows["avg_rmse"] * 10000.0
    table = target_rows.pivot_table(
        index="target",
        columns=["protocol", "model_type"],
        values="rmse_bps",
        aggfunc="mean",
    )
    ordered = [target for target in TARGET_ORDER if target in table.index]
    return table.loc[ordered + [idx for idx in table.index if idx not in ordered]].round(2)


def _fold_rmse_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    fold = fold_metrics.groupby(
        ["fold_id", "validation_start_date", "validation_end_date", "protocol", "model_type"],
        as_index=False,
    )["rmse"].mean()
    fold["rmse_bps"] = fold["rmse"] * 10000.0
    table = fold.pivot_table(
        index=["fold_id", "validation_start_date", "validation_end_date"],
        columns=["protocol", "model_type"],
        values="rmse_bps",
        aggfunc="mean",
    )
    return table.round(2)


def _coefficient_stability_summary(coefficient_stability: pd.DataFrame) -> pd.DataFrame:
    if coefficient_stability.empty:
        return pd.DataFrame()
    result = coefficient_stability.groupby(["protocol", "model_type"], as_index=False).agg(
        coefficient_correlation=("coefficient_correlation", "mean"),
        sign_agreement_pct=("sign_agreement", lambda x: float(np.mean(x) * 100.0)),
        top50_overlap_pct=("top_50_feature_overlap", lambda x: float(np.mean(x) * 100.0)),
        nonzero_jaccard=("nonzero_set_jaccard", "mean"),
        normalized_l2_drift=("normalized_l2_drift", "mean"),
    )
    return result.round(4)


def _coefficient_stability_by_target(coefficient_stability: pd.DataFrame) -> pd.DataFrame:
    if coefficient_stability.empty:
        return pd.DataFrame()
    result = coefficient_stability.groupby(["protocol", "model_type", "target"], as_index=False).agg(
        coefficient_correlation=("coefficient_correlation", "mean"),
        top50_overlap_pct=("top_50_feature_overlap", lambda x: float(np.mean(x) * 100.0)),
        normalized_l2_drift=("normalized_l2_drift", "mean"),
    )
    return result.round(4)


def _vintage_summary(vintage_stability: pd.DataFrame) -> pd.DataFrame:
    if vintage_stability.empty:
        return pd.DataFrame()
    vintage = vintage_stability.copy()
    vintage["rmse_bps"] = vintage["rmse"] * 10000.0
    vintage["prediction_drift_mae_bps"] = vintage["prediction_drift_mae_vs_freshest"] * 10000.0
    table = vintage.groupby(["protocol", "model_type", "staleness_rows"]).agg(
        rmse_bps=("rmse_bps", "mean"),
        correlation=("correlation", "mean"),
        directional_accuracy_pct=("directional_accuracy", lambda x: float(np.mean(x) * 100.0)),
        prediction_drift_mae_bps=("prediction_drift_mae_bps", "mean"),
        coefficient_correlation_vs_freshest=("coefficient_correlation_vs_freshest", "mean"),
        coefficient_l2_drift_vs_freshest=("coefficient_l2_drift_vs_freshest", "mean"),
    )
    return table.round(4)


def _freshest_vintage_target_rmse(vintage_stability: pd.DataFrame) -> pd.DataFrame:
    if vintage_stability.empty:
        return pd.DataFrame()
    freshest = vintage_stability.loc[vintage_stability["staleness_rows"].astype(int).eq(0)].copy()
    freshest["rmse_bps"] = freshest["rmse"] * 10000.0
    table = freshest.pivot_table(
        index="target",
        columns=["protocol", "model_type"],
        values="rmse_bps",
        aggfunc="mean",
    )
    ordered = [target for target in TARGET_ORDER if target in table.index]
    return table.loc[ordered + [idx for idx in table.index if idx not in ordered]].round(2)


def _top_feature_table(coefficient_table: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if coefficient_table.empty:
        return pd.DataFrame()
    rows = []
    for (protocol, model_type, target), group in coefficient_table.groupby(["protocol", "model_type", "target"]):
        top = group.groupby("feature")["abs_coefficient"].mean().sort_values(ascending=False).head(top_n)
        for rank, (feature, value) in enumerate(top.items(), start=1):
            rows.append(
                {
                    "protocol": protocol,
                    "model_type": model_type,
                    "target": target,
                    "rank": rank,
                    "feature": feature,
                    "mean_abs_standardized_coefficient": float(value),
                }
            )
    return pd.DataFrame(rows).round({"mean_abs_standardized_coefficient": 8})


def _embedded_figures(run_dir: Path) -> str:
    figures = [
        ("Fold RMSE over time", run_dir / "figures" / "fold_rmse_over_time.svg"),
        ("Coefficient stability over folds", run_dir / "figures" / "coefficient_stability_over_folds.svg"),
        ("Top-feature coefficient paths", run_dir / "figures" / "top_feature_coefficient_paths.svg"),
        ("Holdout RMSE vs vintage age", run_dir / "figures" / "vintage_holdout_rmse_vs_staleness.svg"),
    ]
    parts = []
    for title, path in figures:
        if path.exists():
            svg = path.read_text(encoding="utf-8")
        else:
            svg = f"<p>Missing figure: {escape(str(path))}</p>"
        parts.append(f'<article class="figure-box"><h3>{escape(title)}</h3>{svg}</article>')
    return "\n".join(parts)


def _html_table(df: pd.DataFrame, *, max_rows: int) -> str:
    if df.empty:
        return "<p>No rows available.</p>"
    display = df.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "".join(f"<th>{escape(str(column))}</th>" for column in display.columns)
    body_rows = []
    for _, row in display.iterrows():
        cells = []
        for value in row:
            cls = "num" if _looks_numeric(value) else ""
            cells.append(f'<td class="{cls}">{escape(str(value))}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    overflow_note = f"<p><em>Showing first {max_rows} of {len(df)} rows.</em></p>" if len(df) > max_rows else ""
    return f'<div class="table-scroll"><table><thead><tr>{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>{overflow_note}'


def _looks_numeric(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
