from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ff5_predictor.io import ensure_dir
from ff5_predictor.nowcast_io import write_json


def write_training_diagnostics(run_dir: Path, training_history: pd.DataFrame) -> dict[str, Any]:
    """Persist neural model training metrics and lightweight SVG training curves."""
    if training_history.empty:
        return {"enabled": False, "n_rows": 0, "paths": {}}

    diagnostics_dir = ensure_dir(run_dir / "training")
    history = training_history.copy()
    history = history.sort_values(["model_type", "cutoff_date", "epoch"]).reset_index(drop=True)

    combined_path = diagnostics_dir / "neural_training_history.csv"
    history.to_csv(combined_path, index=False)

    paths: dict[str, Any] = {"combined_history": str(combined_path), "by_model": {}}
    for model_type, group in history.groupby("model_type", sort=True):
        safe_name = _safe_filename(str(model_type))
        model_history_path = diagnostics_dir / f"{safe_name}_training_history.csv"
        model_curve_path = diagnostics_dir / f"{safe_name}_training_curves.svg"
        group.to_csv(model_history_path, index=False)
        model_curve_path.write_text(_training_curve_svg(str(model_type), group), encoding="utf-8")
        paths["by_model"][str(model_type)] = {
            "history": str(model_history_path),
            "curves": str(model_curve_path),
        }

    metadata = {
        "enabled": True,
        "n_rows": int(len(history)),
        "models": sorted(str(value) for value in history["model_type"].dropna().unique().tolist()),
        "paths": paths,
        "metrics": [
            "train_loss",
            "validation_loss",
            "train_rmse",
            "validation_rmse",
            "train_mae",
            "validation_mae",
            "train_directional_accuracy",
            "validation_directional_accuracy",
        ],
    }
    write_json(diagnostics_dir / "training_diagnostics_metadata.json", metadata)
    return metadata


def _training_curve_svg(model_type: str, history: pd.DataFrame) -> str:
    numeric = history.copy()
    for column in [
        "epoch",
        "train_loss",
        "validation_loss",
        "train_rmse",
        "validation_rmse",
        "train_directional_accuracy",
        "validation_directional_accuracy",
    ]:
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    aggregate = (
        numeric.groupby("epoch", as_index=False)[
            [
                "train_loss",
                "validation_loss",
                "train_rmse",
                "validation_rmse",
                "train_directional_accuracy",
                "validation_directional_accuracy",
            ]
        ]
        .mean(numeric_only=True)
        .sort_values("epoch")
    )

    panels = [
        ("MSE Loss", [("train_loss", "#1f77b4", "train"), ("validation_loss", "#d62728", "validation")]),
        ("RMSE", [("train_rmse", "#1f77b4", "train"), ("validation_rmse", "#d62728", "validation")]),
        (
            "Directional Accuracy",
            [
                ("train_directional_accuracy", "#1f77b4", "train"),
                ("validation_directional_accuracy", "#d62728", "validation"),
            ],
        ),
    ]

    width = 960
    panel_height = 230
    top = 70
    height = top + panel_height * len(panels) + 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#1f2933}.title{font-size:22px;font-weight:700}.label{font-size:13px}.axis{stroke:#96a1ad;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:2.2}.legend{font-size:12px}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="28" y="36" class="title">{escape(model_type)} training curves</text>',
        f'<text x="28" y="58" class="label">Mean curve by epoch across release-gap cutoffs. CSV files contain per-cutoff values.</text>',
    ]

    for index, (title, series) in enumerate(panels):
        panel_top = top + index * panel_height
        parts.extend(_panel_svg(aggregate, title, series, 28, panel_top, width - 56, panel_height - 34))

    parts.append("</svg>")
    return "\n".join(parts)


def _panel_svg(
    data: pd.DataFrame,
    title: str,
    series: list[tuple[str, str, str]],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> list[str]:
    plot_left = x0 + 60
    plot_top = y0 + 28
    plot_width = width - 84
    plot_height = height - 58
    epochs = data["epoch"].to_numpy(dtype=float) if "epoch" in data else np.asarray([], dtype=float)
    value_arrays = [data[column].dropna().to_numpy(dtype=float) for column, _, _ in series if column in data]
    all_values = np.concatenate([values for values in value_arrays if len(values)]) if any(len(v) for v in value_arrays) else np.asarray([0.0, 1.0])
    y_min = float(np.nanmin(all_values))
    y_max = float(np.nanmax(all_values))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = 0.0, 1.0
    if y_min == y_max:
        padding = abs(y_min) * 0.05 or 1.0
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

    x_min = float(np.nanmin(epochs)) if len(epochs) else 1.0
    x_max = float(np.nanmax(epochs)) if len(epochs) else 1.0
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0

    parts = [
        f'<text x="{x0}" y="{y0 + 16}" class="label" font-weight="700">{escape(title)}</text>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" class="axis"/>',
        f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" class="axis"/>',
    ]
    for tick in range(5):
        y = plot_top + (tick / 4) * plot_height
        value = y_max - (tick / 4) * (y_max - y_min)
        parts.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{plot_left - 8}" y="{y + 4:.2f}" text-anchor="end" class="legend">{value:.4g}</text>')
    for idx, (column, color, label) in enumerate(series):
        points = _series_points(data, column, x_min, x_max, y_min, y_max, plot_left, plot_top, plot_width, plot_height)
        if len(points) >= 2:
            path = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(points))
            parts.append(f'<path d="{path}" stroke="{color}" class="line"/>')
        elif len(points) == 1:
            x, y = points[0]
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
        legend_x = plot_left + plot_width - 170 + idx * 86
        parts.append(f'<line x1="{legend_x}" y1="{y0 + 14}" x2="{legend_x + 22}" y2="{y0 + 14}" stroke="{color}" class="line"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{y0 + 18}" class="legend">{escape(label)}</text>')
    parts.append(f'<text x="{plot_left + plot_width}" y="{plot_top + plot_height + 24}" text-anchor="end" class="legend">epoch</text>')
    return parts


def _series_points(
    data: pd.DataFrame,
    column: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    plot_left: int,
    plot_top: int,
    plot_width: int,
    plot_height: int,
) -> list[tuple[float, float]]:
    if column not in data:
        return []
    rows = data[["epoch", column]].dropna()
    points: list[tuple[float, float]] = []
    for _, row in rows.iterrows():
        x_frac = (float(row["epoch"]) - x_min) / (x_max - x_min)
        y_frac = (float(row[column]) - y_min) / (y_max - y_min)
        x = plot_left + x_frac * plot_width
        y = plot_top + (1.0 - y_frac) * plot_height
        points.append((x, y))
    return points


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "model"
