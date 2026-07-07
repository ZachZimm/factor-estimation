from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ff5_predictor.data_yfinance import load_market_data
from ff5_predictor.io import ensure_dir, normalize_datetime_index
from ff5_predictor.nowcast_io import write_json


@dataclass(frozen=True)
class ResidualAnalysisResult:
    run_dir: Path
    residual_panel: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, Any]


def run_residual_analysis(
    config: dict[str, Any],
    *,
    predictions_csv: str | Path | None = None,
    run_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_type: str | None = None,
    release_gap_size: int | None = None,
    gap_day: int | None = None,
) -> ResidualAnalysisResult:
    predictions_path = resolve_residual_predictions_path(predictions_csv=predictions_csv, run_dir=run_dir)
    predictions = pd.read_csv(predictions_path)
    market = load_market_data(config)
    base_run_dir = _default_output_base(predictions_path, run_dir)
    final_output_dir = Path(output_dir) if output_dir is not None else base_run_dir / "analysis" / "residuals"
    return run_residual_analysis_from_frames(
        predictions,
        market,
        config,
        output_dir=final_output_dir,
        model_type=model_type,
        release_gap_size=release_gap_size,
        gap_day=gap_day,
        source_predictions_path=predictions_path,
    )


def run_residual_analysis_from_frames(
    predictions: pd.DataFrame,
    market_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    model_type: str | None = None,
    release_gap_size: int | None = None,
    gap_day: int | None = None,
    source_predictions_path: str | Path | None = None,
) -> ResidualAnalysisResult:
    target_columns = _available_target_columns(predictions, list(config["prediction"]["target_columns"]))
    if not target_columns:
        raise ValueError("Predictions must include matching pred_* and actual_* columns")

    residual_panel = build_residual_panel(
        predictions,
        target_columns,
        config,
        model_type=model_type,
        release_gap_size=release_gap_size,
        gap_day=gap_day,
    )
    market_context = build_market_context(market_df)
    regimes = build_regime_labels(market_context, config)

    summary = residual_summary(residual_panel)
    cross_corr = residual_cross_correlations(residual_panel)
    autocorr = residual_autocorrelations(residual_panel, _int_list(config.get("residual_analysis", {}).get("autocorrelation_lags", [1, 2, 5, 10, 21, 63])))
    market_corr = residual_market_correlations(
        residual_panel,
        market_context,
        _int_list(config.get("residual_analysis", {}).get("market_lags", [-21, -10, -5, -1, 0, 1, 5, 10, 21])),
    )
    regime_summary = residual_regime_summary(residual_panel, regimes)
    regression = residual_market_regressions(residual_panel, market_context)

    output = ensure_dir(output_dir)
    tables_dir = ensure_dir(output / "tables")
    figures_dir = ensure_dir(output / "figures")
    residual_panel.to_csv(tables_dir / "residual_panel.csv", index=False)
    summary.to_csv(tables_dir / "residual_summary.csv", index=False)
    cross_corr.to_csv(tables_dir / "residual_cross_correlation.csv", index=False)
    autocorr.to_csv(tables_dir / "residual_autocorrelation.csv", index=False)
    market_corr.to_csv(tables_dir / "residual_market_lead_lag_correlation.csv", index=False)
    regime_summary.to_csv(tables_dir / "residual_regime_summary.csv", index=False)
    regression.to_csv(tables_dir / "residual_market_regression.csv", index=False)
    market_context.reset_index(names="date").to_csv(tables_dir / "market_context.csv", index=False)
    regimes.reset_index(names="date").to_csv(tables_dir / "regime_labels.csv", index=False)

    figure_paths = write_residual_figures(residual_panel, market_df, market_context, cross_corr, figures_dir)
    metadata = {
        "source_predictions_path": None if source_predictions_path is None else str(source_predictions_path),
        "target_columns": target_columns,
        "model_type_filter": model_type,
        "release_gap_size_filter": release_gap_size,
        "gap_day_filter": gap_day,
        "date_filter": dict(config.get("date_filter", {})),
        "residual_sign_convention": {
            "official_minus_model_implied": "actual - pred",
            "model_implied_minus_official": "pred - actual",
        },
        "market_lag_convention": "positive lag means residual at date t is correlated with market feature at t + lag",
        "n_residual_rows": int(len(residual_panel)),
        "n_models": int(residual_panel["model_type"].nunique()) if not residual_panel.empty else 0,
        "n_dates": int(residual_panel["date"].nunique()) if not residual_panel.empty else 0,
        "table_paths": {
            "residual_panel": str(tables_dir / "residual_panel.csv"),
            "residual_summary": str(tables_dir / "residual_summary.csv"),
            "residual_cross_correlation": str(tables_dir / "residual_cross_correlation.csv"),
            "residual_autocorrelation": str(tables_dir / "residual_autocorrelation.csv"),
            "residual_market_lead_lag_correlation": str(tables_dir / "residual_market_lead_lag_correlation.csv"),
            "residual_regime_summary": str(tables_dir / "residual_regime_summary.csv"),
            "residual_market_regression": str(tables_dir / "residual_market_regression.csv"),
            "market_context": str(tables_dir / "market_context.csv"),
            "regime_labels": str(tables_dir / "regime_labels.csv"),
        },
        "figure_paths": {key: str(path) for key, path in figure_paths.items()},
    }
    write_json(output / "metadata.json", metadata)
    return ResidualAnalysisResult(run_dir=output, residual_panel=residual_panel, summary=summary, metadata=metadata)


def resolve_residual_predictions_path(
    *,
    predictions_csv: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> Path:
    if predictions_csv is not None:
        path = Path(predictions_csv)
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {path}")
        return path
    if run_dir is None:
        raise ValueError("Either predictions_csv or run_dir is required for residual analysis")
    predictions_dir = Path(run_dir) / "predictions"
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"No predictions directory under run dir: {predictions_dir}")
    preferred = (
        "model_implied_factor_series.csv",
        "model_implied_ff5_series.csv",
        "release_gap_predictions.csv",
    )
    for name in preferred:
        path = predictions_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No residual-compatible prediction CSV found in {predictions_dir}")


def build_residual_panel(
    predictions: pd.DataFrame,
    target_columns: list[str],
    config: dict[str, Any],
    *,
    model_type: str | None = None,
    release_gap_size: int | None = None,
    gap_day: int | None = None,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "model_type",
                "target",
                "official",
                "model_implied",
                "official_minus_model_implied",
                "model_implied_minus_official",
                "abs_residual",
            ]
        )
    df = predictions.copy()
    date_column = "target_date" if "target_date" in df.columns else "date"
    df["date"] = pd.to_datetime(df[date_column]).dt.normalize()
    if "model_type" not in df.columns:
        df["model_type"] = "model"
    df["model_type"] = df["model_type"].astype(str)
    if model_type is not None:
        df = df.loc[df["model_type"] == model_type].copy()
    if release_gap_size is not None and "release_gap_size" in df.columns:
        df = df.loc[pd.to_numeric(df["release_gap_size"], errors="coerce") == release_gap_size].copy()
    if gap_day is not None and "gap_day" in df.columns:
        df = df.loc[pd.to_numeric(df["gap_day"], errors="coerce") == gap_day].copy()
    date_filter = config.get("date_filter", {})
    if date_filter.get("start_date"):
        df = df.loc[df["date"] >= pd.Timestamp(date_filter["start_date"])].copy()
    if date_filter.get("end_date"):
        df = df.loc[df["date"] <= pd.Timestamp(date_filter["end_date"])].copy()

    rows: list[pd.DataFrame] = []
    passthrough = [
        column
        for column in ["cutoff_date", "target_date", "gap_day", "release_gap_size", "market_data_asof", "factor_data_asof"]
        if column in df.columns
    ]
    for target in target_columns:
        item = pd.DataFrame(
            {
                "date": df["date"],
                "model_type": df["model_type"],
                "target": target,
                "official": pd.to_numeric(df[f"actual_{target}"], errors="coerce"),
                "model_implied": pd.to_numeric(df[f"pred_{target}"], errors="coerce"),
            }
        )
        for column in passthrough:
            item[column] = df[column].to_numpy()
        item["official_minus_model_implied"] = item["official"] - item["model_implied"]
        item["model_implied_minus_official"] = item["model_implied"] - item["official"]
        item["abs_residual"] = item["official_minus_model_implied"].abs()
        rows.append(item)
    panel = pd.concat(rows, ignore_index=True)
    return panel.dropna(subset=["official", "model_implied", "official_minus_model_implied"]).sort_values(
        ["model_type", "target", "date"]
    )


def build_market_context(market_df: pd.DataFrame) -> pd.DataFrame:
    market = normalize_datetime_index(market_df)
    context = pd.DataFrame(index=market.index)
    if "SPY_close" in market.columns:
        spy = market["SPY_close"].astype(float)
        context["SPY_ret_1d"] = spy.pct_change()
        context["SPY_ret_5d"] = spy.pct_change(5)
        context["SPY_ret_21d"] = spy.pct_change(21)
        context["SPY_ret_63d"] = spy.pct_change(63)
        context["SPY_vol_21d"] = context["SPY_ret_1d"].rolling(21).std()
        context["SPY_vol_63d"] = context["SPY_ret_1d"].rolling(63).std()
    if "^VIX_close" in market.columns:
        vix = market["^VIX_close"].astype(float)
        context["VIX_level"] = vix
        context["VIX_change_1d"] = vix.diff()
        context["VIX_ret_1d"] = vix.pct_change()
    for ticker in ["TLT", "IEF", "SHY", "HYG", "LQD", "UUP", "GLD"]:
        close_col = f"{ticker}_close"
        if close_col in market.columns:
            close = market[close_col].astype(float)
            context[f"{ticker}_ret_1d"] = close.pct_change()
            context[f"{ticker}_ret_21d"] = close.pct_change(21)
    if {"HYG_ret_1d", "LQD_ret_1d"}.issubset(context.columns):
        context["credit_HYG_minus_LQD_ret_1d"] = context["HYG_ret_1d"] - context["LQD_ret_1d"]
    if {"HYG_ret_21d", "LQD_ret_21d"}.issubset(context.columns):
        context["credit_HYG_minus_LQD_ret_21d"] = context["HYG_ret_21d"] - context["LQD_ret_21d"]
    return context.replace([np.inf, -np.inf], np.nan)


def build_regime_labels(market_context: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    regimes = pd.DataFrame(index=market_context.index)
    cfg = config.get("residual_analysis", {})
    high_q = float(cfg.get("high_vol_quantile", 0.8))
    low_q = float(cfg.get("low_vol_quantile", 0.2))
    if "SPY_vol_21d" in market_context.columns:
        vol = market_context["SPY_vol_21d"]
        high = vol.quantile(high_q)
        low = vol.quantile(low_q)
        regimes["volatility_regime"] = np.select([vol >= high, vol <= low], ["high_vol", "low_vol"], default="normal_vol")
    if "SPY_ret_21d" in market_context.columns:
        ret = market_context["SPY_ret_21d"]
        regimes["market_21d_regime"] = np.where(ret >= 0, "market_up_21d", "market_down_21d")
    if "VIX_level" in market_context.columns:
        vix = market_context["VIX_level"]
        high = vix.quantile(high_q)
        low = vix.quantile(low_q)
        regimes["vix_regime"] = np.select([vix >= high, vix <= low], ["high_vix", "low_vix"], default="normal_vix")
    regimes = regimes.astype("object")
    return regimes.where(pd.notna(regimes), None)


def residual_summary(residual_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_type, target), group in residual_panel.groupby(["model_type", "target"], sort=True):
        residual = group["official_minus_model_implied"].astype(float)
        rows.append(
            {
                "model_type": model_type,
                "target": target,
                "n": int(len(residual)),
                "mean_residual": float(residual.mean()),
                "median_residual": float(residual.median()),
                "std_residual": float(residual.std(ddof=0)),
                "mae": float(residual.abs().mean()),
                "rmse": float(np.sqrt(np.mean(residual * residual))),
                "q05": float(residual.quantile(0.05)),
                "q95": float(residual.quantile(0.95)),
                "positive_residual_fraction": float((residual > 0).mean()),
                "lag1_autocorrelation": _safe_corr(residual, residual.shift(1)),
            }
        )
    return pd.DataFrame(rows)


def residual_cross_correlations(residual_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_type, pivot in _residual_pivots(residual_panel).items():
        corr = pivot.corr()
        for left in corr.columns:
            for right in corr.columns:
                rows.append(
                    {
                        "model_type": model_type,
                        "target_x": left,
                        "target_y": right,
                        "correlation": None if pd.isna(corr.loc[left, right]) else float(corr.loc[left, right]),
                    }
                )
    return pd.DataFrame(rows)


def residual_autocorrelations(residual_panel: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    rows = []
    for model_type, pivot in _residual_pivots(residual_panel).items():
        for target in pivot.columns:
            series = pivot[target]
            for lag in lags:
                rows.append(
                    {
                        "model_type": model_type,
                        "target": target,
                        "lag_rows": int(lag),
                        "autocorrelation": _safe_corr(series, series.shift(lag)),
                    }
                )
    return pd.DataFrame(rows)


def residual_market_correlations(residual_panel: pd.DataFrame, market_context: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    context = market_context.dropna(axis=1, how="all")
    rows = []
    for model_type, pivot in _residual_pivots(residual_panel).items():
        aligned = pivot.join(context, how="inner")
        for target in pivot.columns:
            residual = aligned[target]
            for feature in context.columns:
                feature_series = aligned[feature]
                for lag in lags:
                    shifted = feature_series.shift(-lag)
                    rows.append(
                        {
                            "model_type": model_type,
                            "target": target,
                            "market_feature": feature,
                            "market_lag_rows": int(lag),
                            "correlation": _safe_corr(residual, shifted),
                        }
                    )
    return pd.DataFrame(rows)


def residual_regime_summary(residual_panel: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    if regimes.empty:
        return pd.DataFrame(columns=["model_type", "target", "regime_family", "regime", "n", "mean_residual", "mae", "rmse"])
    panel = residual_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    joined = panel.merge(regimes.reset_index(names="date"), on="date", how="left")
    rows = []
    for family in regimes.columns:
        for (model_type, target, regime), group in joined.dropna(subset=[family]).groupby(["model_type", "target", family], sort=True):
            residual = group["official_minus_model_implied"].astype(float)
            rows.append(
                {
                    "model_type": model_type,
                    "target": target,
                    "regime_family": family,
                    "regime": regime,
                    "n": int(len(residual)),
                    "mean_residual": float(residual.mean()),
                    "std_residual": float(residual.std(ddof=0)),
                    "mae": float(residual.abs().mean()),
                    "rmse": float(np.sqrt(np.mean(residual * residual))),
                }
            )
    return pd.DataFrame(rows)


def residual_market_regressions(residual_panel: pd.DataFrame, market_context: pd.DataFrame) -> pd.DataFrame:
    candidate_features = [
        "SPY_ret_1d",
        "SPY_ret_5d",
        "SPY_ret_21d",
        "SPY_vol_21d",
        "VIX_change_1d",
        "VIX_ret_1d",
        "TLT_ret_1d",
        "credit_HYG_minus_LQD_ret_1d",
    ]
    features = [feature for feature in candidate_features if feature in market_context.columns]
    if not features:
        return pd.DataFrame(columns=["model_type", "target", "term", "coefficient", "r2", "n"])
    rows = []
    context = market_context[features]
    for model_type, pivot in _residual_pivots(residual_panel).items():
        aligned = pivot.join(context, how="inner")
        for target in pivot.columns:
            data = aligned[[target, *features]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) < max(8, len(features) + 2):
                continue
            y = data[target].to_numpy(dtype=float)
            X = data[features].to_numpy(dtype=float)
            X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
            design = np.column_stack([np.ones(len(X)), X])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            pred = design @ coef
            total = float(np.sum((y - y.mean()) ** 2))
            r2 = None if total == 0.0 else float(1.0 - np.sum((y - pred) ** 2) / total)
            rows.append({"model_type": model_type, "target": target, "term": "intercept", "coefficient": float(coef[0]), "r2": r2, "n": int(len(data))})
            for feature, value in zip(features, coef[1:]):
                rows.append({"model_type": model_type, "target": target, "term": feature, "coefficient": float(value), "r2": r2, "n": int(len(data))})
    return pd.DataFrame(rows)


def write_residual_figures(
    residual_panel: pd.DataFrame,
    market_df: pd.DataFrame,
    market_context: pd.DataFrame,
    cross_corr: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_type, pivot in _residual_pivots(residual_panel).items():
        safe = _safe_filename(model_type)
        path = figures_dir / f"{safe}_residual_timeseries.svg"
        path.write_text(_line_svg(f"{model_type}: official minus model-implied residuals", pivot), encoding="utf-8")
        paths[f"{model_type}_residual_timeseries"] = path
        overlay_path = figures_dir / f"{safe}_residual_market_overlay.svg"
        overlay_path.write_text(
            _spy_performance_residual_svg(
                f"{model_type}: absolute residuals and SPY performance",
                pivot,
                market_df,
            ),
            encoding="utf-8",
        )
        paths[f"{model_type}_residual_market_overlay"] = overlay_path
        model_corr = cross_corr.loc[cross_corr["model_type"] == model_type]
        if not model_corr.empty:
            heatmap_path = figures_dir / f"{safe}_residual_cross_correlation.svg"
            heatmap_path.write_text(_heatmap_svg(f"{model_type}: residual cross-correlation", model_corr), encoding="utf-8")
            paths[f"{model_type}_residual_cross_correlation"] = heatmap_path
    return paths


def _available_target_columns(predictions: pd.DataFrame, configured_targets: list[str]) -> list[str]:
    return [target for target in configured_targets if f"pred_{target}" in predictions.columns and f"actual_{target}" in predictions.columns]


def _default_output_base(predictions_path: Path, run_dir: str | Path | None) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    if predictions_path.parent.name == "predictions":
        return predictions_path.parent.parent
    return predictions_path.parent


def _residual_pivots(residual_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pivots: dict[str, pd.DataFrame] = {}
    if residual_panel.empty:
        return pivots
    panel = residual_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    for model_type, group in panel.groupby("model_type", sort=True):
        pivot = group.pivot_table(
            index="date",
            columns="target",
            values="official_minus_model_implied",
            aggfunc="mean",
        ).sort_index()
        pivots[str(model_type)] = pivot
    return pivots


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return values * np.nan
    return (values - values.mean()) / std


def _safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    data = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(data) < 2:
        return None
    if data.iloc[:, 0].std(ddof=0) == 0 or data.iloc[:, 1].std(ddof=0) == 0:
        return None
    value = data.iloc[:, 0].corr(data.iloc[:, 1])
    return None if pd.isna(value) else float(value)


def _int_list(values: list[Any]) -> list[int]:
    return [int(value) for value in values]


def _line_svg(title: str, data: pd.DataFrame) -> str:
    data = data.dropna(how="all")
    width, height = 980, 420
    margin = {"left": 70, "right": 28, "top": 58, "bottom": 42}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#7f7f7f"]
    if data.empty:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="40">{escape(title)}: no data</text></svg>'
    x_values = pd.to_datetime(data.index)
    x_min = x_values.min().value
    x_max = x_values.max().value
    y_values = data.to_numpy(dtype=float)
    finite = y_values[np.isfinite(y_values)]
    y_min = float(finite.min()) if len(finite) else -1.0
    y_max = float(finite.max()) if len(finite) else 1.0
    if y_min == y_max:
        pad = abs(y_min) * 0.1 or 1.0
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad
    parts = [_svg_header(width, height), f'<text x="24" y="34" class="title">{escape(title)}</text>']
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" class="axis"/>')
    for tick in range(5):
        y = margin["top"] + tick / 4 * plot_h
        value = y_max - tick / 4 * (y_max - y_min)
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" class="legend">{value:.3g}</text>')
    for index, column in enumerate(data.columns):
        points = []
        for date, value in data[column].dropna().items():
            x_frac = 0.5 if x_min == x_max else (pd.Timestamp(date).value - x_min) / (x_max - x_min)
            y_frac = (float(value) - y_min) / (y_max - y_min)
            points.append((margin["left"] + x_frac * plot_w, margin["top"] + (1.0 - y_frac) * plot_h))
        color = colors[index % len(colors)]
        if len(points) >= 2:
            path = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(points))
            parts.append(f'<path d="{path}" stroke="{color}" class="line"/>')
        legend_x = margin["left"] + 12 + (index % 4) * 210
        legend_y = height - 18 - (index // 4) * 18
        parts.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 24}" y2="{legend_y - 4}" stroke="{color}" class="line"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{legend_y}" class="legend">{escape(str(column))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _spy_performance_residual_svg(title: str, residual_pivot: pd.DataFrame, market_df: pd.DataFrame) -> str:
    market = normalize_datetime_index(market_df)
    required = ["SPY_close"]
    width, height = 1120, 560
    if residual_pivot.empty or not all(column in market.columns for column in required):
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="40">{escape(title)}: SPY close data unavailable</text></svg>'

    residual = pd.DataFrame(index=residual_pivot.index)
    residual["mean_abs_residual_bps"] = residual_pivot.abs().mean(axis=1) * 10000.0
    residual["max_abs_residual_bps"] = residual_pivot.abs().max(axis=1) * 10000.0
    start = pd.Timestamp(residual.index.min())
    end = pd.Timestamp(residual.index.max())
    spy = market.loc[start:end, required].dropna(how="any")
    if spy.empty:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="40">{escape(title)}: no overlapping SPY close data</text></svg>'
    residual = residual.reindex(spy.index).dropna(how="all")
    if residual.empty:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="40">{escape(title)}: no overlapping residual data</text></svg>'

    margin = {"left": 88, "right": 104, "top": 92, "bottom": 72}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    dates = pd.DatetimeIndex(spy.index)
    date_positions = {pd.Timestamp(date).normalize(): idx for idx, date in enumerate(dates)}
    x_denominator = max(1, len(dates) - 1)

    spy_close = spy["SPY_close"].astype(float)
    performance = np.log(spy_close / spy_close.iloc[0])
    performance_min = float(np.nanmin(performance.to_numpy(dtype=float)))
    performance_max = float(np.nanmax(performance.to_numpy(dtype=float)))
    if performance_min == performance_max:
        performance_pad = abs(performance_min) * 0.1 or 0.01
    else:
        performance_pad = (performance_max - performance_min) * 0.08
    performance_min -= performance_pad
    performance_max += performance_pad

    residual_max = float(np.nanmax(residual.to_numpy(dtype=float)))
    residual_max = residual_max * 1.12 if residual_max > 0 else 1.0

    def x_for(date: pd.Timestamp) -> float:
        position = date_positions[pd.Timestamp(date).normalize()]
        return margin["left"] + (position / x_denominator) * plot_w

    def left_y(value: float) -> float:
        return margin["top"] + (1.0 - (value - performance_min) / (performance_max - performance_min)) * plot_h

    def right_y(value: float) -> float:
        return margin["top"] + (1.0 - value / residual_max) * plot_h

    parts = [
        _svg_header(width, height),
        f'<text x="24" y="34" class="title">{escape(title)}</text>',
        '<text x="24" y="56" class="legend">Single-pane overlay: SPY cumulative return uses the left log-return axis; absolute residuals are raw observed values on the right bps axis.</text>',
    ]
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{margin["left"] + plot_w}" y1="{margin["top"]}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" class="axis"/>')
    parts.append(f'<text x="{margin["left"]}" y="{margin["top"] - 18}" class="legend">SPY cumulative return, log scale</text>')
    parts.append(f'<text x="{margin["left"] + plot_w}" y="{margin["top"] - 18}" text-anchor="end" class="legend">Observed abs residual, bps</text>')

    for tick in range(5):
        y = margin["top"] + (tick / 4) * plot_h
        log_value = performance_max - (tick / 4) * (performance_max - performance_min)
        return_label = (np.exp(log_value) - 1.0) * 100.0
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{margin["left"] - 10}" y="{y + 4:.2f}" text-anchor="end" class="legend">{return_label:.1f}%</text>')
    for tick in range(5):
        y = margin["top"] + (tick / 4) * plot_h
        value = residual_max - (tick / 4) * residual_max
        parts.append(f'<text x="{margin["left"] + plot_w + 10}" y="{y + 4:.2f}" class="legend">{value:.1f}</text>')

    if performance_min < 0 < performance_max:
        zero_y = left_y(0.0)
        parts.append(f'<line x1="{margin["left"]}" y1="{zero_y:.2f}" x2="{margin["left"] + plot_w}" y2="{zero_y:.2f}" stroke="#8b9bab" stroke-width="1.4" stroke-dasharray="4 5"/>')

    performance_points = [(x_for(pd.Timestamp(date)), left_y(float(value))) for date, value in performance.dropna().items()]
    if len(performance_points) >= 2:
        path = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(performance_points))
        parts.append(f'<path d="{path}" fill="none" stroke="#263b59" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/>')

    residual_colors = {
        "mean_abs_residual_bps": "#174a5a",
        "max_abs_residual_bps": "#b46732",
    }
    for column, color in residual_colors.items():
        points = []
        for date, value in residual[column].dropna().items():
            points.append((x_for(pd.Timestamp(date)), right_y(float(value))))
        if len(points) >= 2:
            path = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(points))
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" opacity="0.82"/>')

    legend_y = height - 18
    legend_items = [
        ("SPY cumulative return, log-scaled", "#263b59"),
        ("mean abs residual, observed", "#174a5a"),
        ("max abs residual, observed", "#b46732"),
    ]
    legend_x = margin["left"]
    for label, color in legend_items:
        parts.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 24}" y2="{legend_y - 4}" stroke="{color}" class="line"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{legend_y}" class="legend">{escape(label)}</text>')
        legend_x += 305

    for idx in np.linspace(0, len(dates) - 1, num=min(6, len(dates)), dtype=int):
        date = dates[idx]
        x = x_for(pd.Timestamp(date))
        parts.append(f'<text x="{x:.2f}" y="{margin["top"] + plot_h + 26}" text-anchor="middle" class="legend">{date.strftime("%Y-%m")}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _heatmap_svg(title: str, corr_long: pd.DataFrame) -> str:
    targets = sorted(set(corr_long["target_x"]).union(set(corr_long["target_y"])))
    cell = 46
    left = 120
    top = 66
    width = left + cell * len(targets) + 40
    height = top + cell * len(targets) + 70
    lookup = {
        (row["target_x"], row["target_y"]): row["correlation"]
        for _, row in corr_long.iterrows()
    }
    parts = [_svg_header(width, height), f'<text x="22" y="34" class="title">{escape(title)}</text>']
    for i, target in enumerate(targets):
        parts.append(f'<text x="{left + i * cell + cell / 2:.1f}" y="{top - 12}" text-anchor="middle" class="legend">{escape(target)}</text>')
        parts.append(f'<text x="{left - 12}" y="{top + i * cell + cell / 2 + 4:.1f}" text-anchor="end" class="legend">{escape(target)}</text>')
    for row_idx, left_target in enumerate(targets):
        for col_idx, right_target in enumerate(targets):
            value = lookup.get((left_target, right_target))
            color = _corr_color(None if pd.isna(value) else float(value))
            x = left + col_idx * cell
            y = top + row_idx * cell
            label = "" if value is None or pd.isna(value) else f"{float(value):.2f}"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4:.1f}" text-anchor="middle" class="legend">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _corr_color(value: float | None) -> str:
    if value is None:
        return "#f3f4f6"
    clipped = max(-1.0, min(1.0, value))
    if clipped >= 0:
        intensity = int(255 - 120 * clipped)
        return f"rgb({intensity},{intensity},255)"
    intensity = int(255 - 120 * abs(clipped))
    return f"rgb(255,{intensity},{intensity})"


def _svg_header(width: int, height: int) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<style>",
            "text{font-family:Inter,Arial,sans-serif;fill:#1f2933}.title{font-size:20px;font-weight:700}.legend{font-size:12px}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:2}",
            "</style>",
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        ]
    )


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "model"
