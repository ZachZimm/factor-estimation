from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ff5_predictor.io import ensure_dir


VIEWER_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
PREFERRED_RUN_PREDICTION_FILES = (
    "release_gap_predictions.csv",
    "latest_nowcast.csv",
)


def filter_ff5_for_viewer(
    ff5_df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    df = ff5_df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    if start_date:
        df = df.loc[pd.Timestamp(start_date) :]
    if end_date:
        df = df.loc[: pd.Timestamp(end_date)]
    columns = [col for col in VIEWER_COLUMNS if col in df.columns]
    return df[columns].dropna(how="all")


def resolve_predictions_path(
    predictions_csv: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> Path | None:
    if predictions_csv is not None:
        path = Path(predictions_csv)
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {path}")
        return path
    if run_dir is None:
        return None
    predictions_dir = Path(run_dir) / "predictions"
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"No predictions directory under run dir: {predictions_dir}")
    for name in PREFERRED_RUN_PREDICTION_FILES:
        path = predictions_dir / name
        if path.exists():
            return path
    csv_files = sorted(predictions_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No prediction CSV files found in {predictions_dir}")
    return csv_files[0]


def load_predictions_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    date_column = "target_date" if "target_date" in df.columns else "date"
    if date_column not in df.columns:
        raise ValueError(f"Predictions file must include 'date' or 'target_date': {path}")
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_column]).dt.tz_localize(None).dt.normalize()
    return df.sort_values("date")


def prepare_predictions_for_viewer(
    predictions: pd.DataFrame,
    target_columns: list[str],
    default_model: str | None = None,
    default_gap_day: int | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    if predictions.empty:
        raise ValueError("Predictions dataframe is empty")

    models = (
        sorted(predictions["model_type"].dropna().astype(str).unique().tolist())
        if "model_type" in predictions.columns
        else ["default"]
    )
    gap_days: list[int] = []
    if "gap_day" in predictions.columns:
        gap_days = sorted(
            int(value)
            for value in predictions["gap_day"].dropna().unique().tolist()
        )

    chosen_model = default_model if default_model in models else models[0]
    chosen_gap_day: int | None = default_gap_day
    if gap_days:
        if chosen_gap_day not in gap_days:
            chosen_gap_day = gap_days[0]

    prediction_rows: list[dict[str, Any]] = []
    for _, row in predictions.iterrows():
        item: dict[str, Any] = {
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "model_type": str(row["model_type"]) if "model_type" in predictions.columns else "default",
        }
        if gap_days:
            gap_value = row.get("gap_day")
            item["gap_day"] = None if pd.isna(gap_value) else int(gap_value)
        for column in target_columns:
            pred_column = f"pred_{column}"
            actual_column = f"actual_{column}"
            if pred_column in predictions.columns:
                pred_value = row[pred_column]
                item[f"{column}_pred"] = None if pd.isna(pred_value) else float(pred_value)
            if actual_column in predictions.columns:
                actual_value = row[actual_column]
                item[f"{column}_actual"] = None if pd.isna(actual_value) else float(actual_value)
        prediction_rows.append(item)

    return {
        "enabled": True,
        "predictionRows": prediction_rows,
        "models": models,
        "gapDays": gap_days,
        "defaultModel": chosen_model,
        "defaultGapDay": chosen_gap_day,
        "runLabel": run_label or "predictions",
        "seriesModes": ["official", "predicted", "both", "error"],
        "defaultSeriesMode": "both",
    }


def write_factor_viewer_html(
    ff5_df: pd.DataFrame,
    output_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    predictions_df: pd.DataFrame | None = None,
    target_columns: list[str] | None = None,
    default_model: str | None = None,
    default_gap_day: int | None = None,
    run_label: str | None = None,
) -> Path:
    filtered = filter_ff5_for_viewer(ff5_df, start_date=start_date, end_date=end_date)
    output = Path(output_path)
    ensure_dir(output.parent)
    factors = target_columns or [column for column in VIEWER_COLUMNS if column in filtered.columns and column != "RF"]
    comparison: dict[str, Any] | None = None
    if predictions_df is not None:
        comparison = prepare_predictions_for_viewer(
            predictions_df,
            factors,
            default_model=default_model,
            default_gap_day=default_gap_day,
            run_label=run_label,
        )
    payload = _viewer_payload(filtered, factors=factors, comparison=comparison)
    html = _viewer_template()
    html = html.replace("__FF5_PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    output.write_text(html, encoding="utf-8")
    return output


def _viewer_payload(
    df: pd.DataFrame,
    factors: list[str] | None = None,
    comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_columns = factors or ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    records: list[dict[str, Any]] = []
    for date, row in df.iterrows():
        item: dict[str, Any] = {"date": date.strftime("%Y-%m-%d")}
        for column in VIEWER_COLUMNS:
            if column in df.columns:
                value = row[column]
                item[column] = None if pd.isna(value) else float(value)
        records.append(item)
    return {
        "columns": [column for column in VIEWER_COLUMNS if column in df.columns],
        "targetColumns": [column for column in target_columns if column in df.columns],
        "records": records,
        "units": "decimal_returns",
        "displayUnits": "percent",
        "comparison": comparison,
    }


def _viewer_template() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FF5 Factor Viewer</title>
  <style>
    :root {
      --ink: #17211d;
      --muted: #66736d;
      --line: #d7ddd9;
      --field: #f7f8f5;
      --paper: #fbfcf8;
      --panel: #ffffff;
      --accent: #0f7a5f;
      --accent-2: #b7472a;
      --shadow: 0 18px 48px rgba(23, 33, 29, 0.1);
      --radius: 8px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(15, 122, 95, 0.05) 1px, transparent 1px),
        linear-gradient(rgba(15, 122, 95, 0.04) 1px, transparent 1px),
        var(--paper);
      background-size: 28px 28px;
      font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
    }

    header {
      padding: 28px clamp(18px, 4vw, 48px) 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(251, 252, 248, 0.92);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 5;
    }

    h1 {
      margin: 0;
      font-size: clamp(24px, 4vw, 42px);
      line-height: 1;
      letter-spacing: 0;
      font-weight: 800;
    }

    .subhead {
      margin-top: 9px;
      max-width: 920px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }

    main {
      padding: 22px clamp(18px, 4vw, 48px) 40px;
      display: grid;
      grid-template-columns: minmax(220px, 300px) minmax(0, 1fr);
      gap: 22px;
      align-items: start;
    }

    .controls,
    .workspace {
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }

    .controls {
      padding: 16px;
      position: sticky;
      top: 118px;
    }

    .workspace {
      min-width: 0;
      overflow: hidden;
    }

    .section-title {
      margin: 0 0 10px;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 700;
    }

    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }

    input[type="date"],
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--field);
      color: var(--ink);
      padding: 10px 9px;
      font: inherit;
      font-size: 13px;
      min-height: 38px;
    }

    .field-row {
      display: grid;
      gap: 10px;
      margin-bottom: 14px;
    }

    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 14px 0 18px;
    }

    button {
      border: 1px solid var(--ink);
      border-radius: 6px;
      background: var(--ink);
      color: #fff;
      min-height: 38px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }

    button.secondary {
      color: var(--ink);
      background: transparent;
      border-color: var(--line);
    }

    button:hover { filter: brightness(0.96); }

    .factor-list {
      display: grid;
      gap: 8px;
      margin-bottom: 18px;
    }

    .factor-toggle {
      display: grid;
      grid-template-columns: 20px 12px 1fr;
      align-items: center;
      gap: 8px;
      min-height: 28px;
      color: var(--ink);
      font-size: 13px;
      margin: 0;
    }

    .factor-toggle input { margin: 0; }

    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      border: 1px solid rgba(23, 33, 29, 0.2);
    }

    .readout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      border-top: 1px solid var(--line);
      padding-top: 12px;
      gap: 10px;
      font-size: 12px;
    }

    .readout strong {
      display: block;
      font-size: 17px;
      color: var(--ink);
      margin-top: 3px;
    }

    .chart-top {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 18px 10px;
      border-bottom: 1px solid var(--line);
    }

    .chart-title {
      margin: 0;
      font-size: 17px;
      line-height: 1.2;
    }

    .chart-meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 5px;
    }

    .canvas-wrap {
      position: relative;
      height: min(62vh, 620px);
      min-height: 390px;
      padding: 12px;
    }

    canvas {
      display: block;
      width: 100%;
      height: 100%;
      background: #fffefa;
      border: 1px solid var(--line);
      border-radius: 6px;
    }

    .tooltip {
      position: absolute;
      pointer-events: none;
      min-width: 190px;
      max-width: 320px;
      padding: 10px 11px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 12px 28px rgba(23, 33, 29, 0.16);
      font-size: 12px;
      display: none;
      z-index: 4;
    }

    .tooltip-date {
      font-weight: 800;
      margin-bottom: 6px;
    }

    .tooltip-line {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      margin-top: 3px;
    }

    .stats {
      border-top: 1px solid var(--line);
      padding: 0 18px 18px;
      overflow: auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      min-width: 680px;
    }

    th,
    td {
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }

    th:first-child,
    td:first-child { text-align: left; }

    th {
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 10px;
    }

    .empty {
      padding: 30px;
      color: var(--muted);
      text-align: center;
    }

    @media (max-width: 900px) {
      header { position: static; }
      main { grid-template-columns: 1fr; }
      .controls { position: static; }
      .canvas-wrap { height: 460px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>FF5 Factor Viewer</h1>
    <div id="pageSubhead" class="subhead">Daily Fama-French 5-factor data displayed in percent units. Source values are cached internally as decimal returns; the chart multiplies by 100 for readability.</div>
  </header>

  <main>
    <aside class="controls">
      <p class="section-title">Date Range</p>
      <div class="field-row">
        <div>
          <label for="startDate">Start</label>
          <input id="startDate" type="date">
        </div>
        <div>
          <label for="endDate">End</label>
          <input id="endDate" type="date">
        </div>
      </div>

      <p class="section-title">Factors</p>
      <div id="factorList" class="factor-list"></div>

      <div id="comparisonControls" hidden>
        <p class="section-title">Run Comparison</p>
        <div class="field-row">
          <div>
            <label for="seriesMode">Series</label>
            <select id="seriesMode">
              <option value="official">Official</option>
              <option value="predicted">Predicted</option>
              <option value="both">Both</option>
              <option value="error">Error (pred − official)</option>
            </select>
          </div>
          <div>
            <label for="modelSelect">Model</label>
            <select id="modelSelect"></select>
          </div>
        </div>
        <div id="gapDayField" class="field-row" hidden>
          <div>
            <label for="gapDaySelect">Gap day</label>
            <select id="gapDaySelect"></select>
          </div>
        </div>
        <div id="runLabel" class="chart-meta" style="margin-bottom:14px"></div>
      </div>

      <p class="section-title">View</p>
      <label class="factor-toggle" for="cumulativeMode">
        <input id="cumulativeMode" type="checkbox">
        <span class="swatch" style="background:var(--ink)"></span>
        <span>Cumulative sum</span>
      </label>

      <div class="button-row">
        <button id="resetButton" class="secondary" type="button">Reset</button>
        <button id="downloadButton" type="button">CSV</button>
      </div>

      <p class="section-title">Window</p>
      <div class="readout">
        <div>Rows<strong id="rowCount">0</strong></div>
        <div>Span<strong id="dateSpan">-</strong></div>
      </div>
    </aside>

    <section class="workspace">
      <div class="chart-top">
        <div>
          <h2 id="chartTitle" class="chart-title">Factor Returns Over Time</h2>
          <div id="chartMeta" class="chart-meta"></div>
        </div>
      </div>
      <div class="canvas-wrap">
        <canvas id="factorChart"></canvas>
        <div id="tooltip" class="tooltip"></div>
      </div>
      <div class="stats">
        <table>
          <thead>
            <tr>
              <th>Factor</th>
              <th>Mean</th>
              <th>Std</th>
              <th>Min</th>
              <th>Max</th>
              <th>Positive</th>
              <th>Latest</th>
            </tr>
          </thead>
          <tbody id="statsBody"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const payload = __FF5_PAYLOAD__;
    const officialRecords = payload.records;
    const comparison = payload.comparison;
    const predictionRows = comparison ? comparison.predictionRows : [];
    const factorColumns = payload.targetColumns.filter((name) => payload.columns.includes(name));
    const colors = {
      "Mkt-RF": "#0f7a5f",
      "SMB": "#b7472a",
      "HML": "#2f5f9f",
      "RMW": "#8b5b12",
      "CMA": "#6f4b8b",
      "RF": "#59636d"
    };
    const state = { active: new Set(factorColumns), hoverIndex: null };

    const startInput = document.getElementById("startDate");
    const endInput = document.getElementById("endDate");
    const cumulativeInput = document.getElementById("cumulativeMode");
    const factorList = document.getElementById("factorList");
    const rowCount = document.getElementById("rowCount");
    const dateSpan = document.getElementById("dateSpan");
    const chartMeta = document.getElementById("chartMeta");
    const chartTitle = document.getElementById("chartTitle");
    const pageSubhead = document.getElementById("pageSubhead");
    const statsBody = document.getElementById("statsBody");
    const canvas = document.getElementById("factorChart");
    const tooltip = document.getElementById("tooltip");
    const comparisonControls = document.getElementById("comparisonControls");
    const seriesModeSelect = document.getElementById("seriesMode");
    const modelSelect = document.getElementById("modelSelect");
    const gapDayField = document.getElementById("gapDayField");
    const gapDaySelect = document.getElementById("gapDaySelect");
    const runLabel = document.getElementById("runLabel");
    const ctx = canvas.getContext("2d");

    const seriesModeTitles = {
      official: "Official FF5 Factors",
      predicted: "Predicted FF5 Factors",
      both: "Official vs Predicted FF5",
      error: "Prediction Error (pred − official)"
    };

    function formatPct(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return `${(value * 100).toFixed(digits)}%`;
    }

    function inDateRange(row, start, end) {
      return (!start || row.date >= start) && (!end || row.date <= end);
    }

    function filteredOfficialRecords() {
      const start = startInput.value;
      const end = endInput.value;
      return officialRecords.filter((row) => inDateRange(row, start, end));
    }

    function filteredPredictionRows() {
      const start = startInput.value;
      const end = endInput.value;
      const model = modelSelect.value;
      const gapDay = gapDaySelect.value;
      return predictionRows.filter((row) => {
        if (comparison && row.model_type !== model) return false;
        if (comparison && comparison.gapDays.length && gapDay !== "all" && String(row.gap_day) !== gapDay) {
          return false;
        }
        return inDateRange(row, start, end);
      });
    }

    function officialMap(rows) {
      return new Map(rows.map((row) => [row.date, row]));
    }

    function baseChartRows() {
      if (!comparison) return filteredOfficialRecords();

      const mode = seriesModeSelect.value;
      const official = filteredOfficialRecords();
      const officialByDate = officialMap(official);
      const preds = filteredPredictionRows();

      if (mode === "official") return official;

      if (mode === "predicted") {
        return preds.map((row) => {
          const chartRow = { date: row.date };
          for (const factor of factorColumns) {
            chartRow[factor] = row[`${factor}_pred`] ?? null;
          }
          return chartRow;
        });
      }

      if (mode === "error") {
        return preds.map((row) => {
          const officialRow = officialByDate.get(row.date);
          const chartRow = { date: row.date };
          for (const factor of factorColumns) {
            const pred = row[`${factor}_pred`];
            const actual = row[`${factor}_actual`] ?? (officialRow ? officialRow[factor] : null);
            chartRow[factor] =
              pred !== null && pred !== undefined && actual !== null && actual !== undefined
                ? pred - actual
                : null;
          }
          return chartRow;
        });
      }

      return preds.map((row) => {
        const officialRow = officialByDate.get(row.date);
        const chartRow = { date: row.date };
        for (const factor of factorColumns) {
          chartRow[factor] = row[`${factor}_actual`] ?? (officialRow ? officialRow[factor] : null);
          chartRow[`${factor}_pred`] = row[`${factor}_pred`] ?? null;
        }
        return chartRow;
      });
    }

    function chartSeriesKeys(factor) {
      if (!comparison || seriesModeSelect.value !== "both") {
        return [{ key: factor, label: factor, dash: [], width: factor === "Mkt-RF" ? 2 : 1.5, alpha: 1 }];
      }
      return [
        { key: factor, label: `${factor} official`, dash: [], width: factor === "Mkt-RF" ? 2 : 1.5, alpha: 1 },
        { key: `${factor}_pred`, label: `${factor} pred`, dash: [6, 4], width: 1.5, alpha: 0.85 }
      ];
    }

    function displayRecords() {
      const rows = baseChartRows();
      if (!cumulativeInput.checked) return rows;

      const keys = new Set();
      for (const factor of factorColumns) {
        for (const series of chartSeriesKeys(factor)) keys.add(series.key);
      }
      const totals = Object.fromEntries(Array.from(keys).map((key) => [key, 0]));

      return rows.map((row) => {
        const cumulative = { date: row.date };
        for (const key of keys) {
          const value = row[key];
          if (value === null || value === undefined || Number.isNaN(value)) {
            cumulative[key] = null;
          } else {
            totals[key] += value;
            cumulative[key] = totals[key];
          }
        }
        return cumulative;
      });
    }

    function initComparisonControls() {
      if (!comparison) return;

      comparisonControls.hidden = false;
      pageSubhead.textContent =
        "Compare official Kenneth French factors against model run predictions. Use the series mode to switch between official, predicted, overlay, and error views.";
      seriesModeSelect.value = comparison.defaultSeriesMode || "both";
      runLabel.textContent = comparison.runLabel;

      modelSelect.innerHTML = "";
      for (const model of comparison.models) {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
      }
      modelSelect.value = comparison.defaultModel;

      if (comparison.gapDays.length) {
        gapDayField.hidden = false;
        gapDaySelect.innerHTML = "";
        for (const gapDay of comparison.gapDays) {
          const option = document.createElement("option");
          option.value = String(gapDay);
          option.textContent = `Gap day ${gapDay}`;
          gapDaySelect.appendChild(option);
        }
        gapDaySelect.value = String(comparison.defaultGapDay ?? comparison.gapDays[0]);
      } else {
        gapDayField.hidden = true;
        gapDaySelect.innerHTML = '<option value="all">All</option>';
        gapDaySelect.value = "all";
      }

      seriesModeSelect.addEventListener("change", draw);
      modelSelect.addEventListener("change", draw);
      gapDaySelect.addEventListener("change", draw);
    }

    function initControls() {
      const dateCandidates = comparison
        ? [...new Set([...officialRecords.map((row) => row.date), ...predictionRows.map((row) => row.date)])].sort()
        : officialRecords.map((row) => row.date);

      if (dateCandidates.length) {
        startInput.min = dateCandidates[0];
        startInput.max = dateCandidates[dateCandidates.length - 1];
        endInput.min = dateCandidates[0];
        endInput.max = dateCandidates[dateCandidates.length - 1];
        startInput.value = dateCandidates[0];
        endInput.value = dateCandidates[dateCandidates.length - 1];
      }

      initComparisonControls();

      factorList.innerHTML = "";
      for (const factor of factorColumns) {
        const label = document.createElement("label");
        label.className = "factor-toggle";
        label.innerHTML = `
          <input type="checkbox" checked data-factor="${factor}">
          <span class="swatch" style="background:${colors[factor]}"></span>
          <span>${factor}</span>
        `;
        factorList.appendChild(label);
      }

      factorList.addEventListener("change", (event) => {
        const input = event.target;
        if (!input.matches("input[type='checkbox']")) return;
        const factor = input.dataset.factor;
        if (input.checked) state.active.add(factor);
        else state.active.delete(factor);
        draw();
      });

      startInput.addEventListener("change", draw);
      endInput.addEventListener("change", draw);
      cumulativeInput.addEventListener("change", draw);
      document.getElementById("resetButton").addEventListener("click", () => {
        if (dateCandidates.length) {
          startInput.value = dateCandidates[0];
          endInput.value = dateCandidates[dateCandidates.length - 1];
        }
        cumulativeInput.checked = false;
        state.active = new Set(factorColumns);
        for (const input of factorList.querySelectorAll("input")) input.checked = true;
        if (comparison) {
          seriesModeSelect.value = comparison.defaultSeriesMode || "both";
          modelSelect.value = comparison.defaultModel;
          if (comparison.gapDays.length) {
            gapDaySelect.value = String(comparison.defaultGapDay ?? comparison.gapDays[0]);
          }
        }
        draw();
      });
      document.getElementById("downloadButton").addEventListener("click", downloadCsv);
      canvas.addEventListener("mousemove", handleHover);
      canvas.addEventListener("mouseleave", () => {
        state.hoverIndex = null;
        tooltip.style.display = "none";
        draw();
      });
      window.addEventListener("resize", draw);
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return rect;
    }

    function plottedSeries(active) {
      const series = [];
      for (const factor of active) {
        for (const item of chartSeriesKeys(factor)) series.push(item);
      }
      return series;
    }

    function extent(rows, active) {
      let min = Infinity;
      let max = -Infinity;
      const series = plottedSeries(active);
      for (const row of rows) {
        for (const item of series) {
          const value = row[item.key];
          if (value === null || value === undefined || Number.isNaN(value)) continue;
          min = Math.min(min, value);
          max = Math.max(max, value);
        }
      }
      if (!Number.isFinite(min) || !Number.isFinite(max)) return [-0.01, 0.01];
      if (min === max) {
        min -= 0.001;
        max += 0.001;
      }
      const pad = (max - min) * 0.08;
      return [min - pad, max + pad];
    }

    function currentSeriesMode() {
      return comparison ? seriesModeSelect.value : "official";
    }

    function draw() {
      const rows = displayRecords();
      const active = Array.from(state.active);
      const seriesMode = currentSeriesMode();
      const modeLabel = cumulativeInput.checked ? "cumulative sum" : "daily returns";
      const rect = resizeCanvas();
      const width = rect.width;
      const height = rect.height;
      ctx.clearRect(0, 0, width, height);

      chartTitle.textContent = seriesModeTitles[seriesMode] || "Factor Returns Over Time";
      rowCount.textContent = rows.length.toLocaleString();
      dateSpan.textContent = rows.length ? `${rows[0].date} to ${rows[rows.length - 1].date}` : "-";
      const comparisonBits = comparison
        ? ` | ${seriesMode} | ${modelSelect.value}${comparison.gapDays.length ? ` | gap ${gapDaySelect.value}` : ""}`
        : "";
      chartMeta.textContent = rows.length
        ? `${rows.length.toLocaleString()} rows | ${modeLabel}${comparisonBits} | ${active.join(", ") || "no factors selected"}`
        : "No data in selected range";
      updateStats(rows, active);

      if (!rows.length || !active.length) {
        ctx.fillStyle = "#66736d";
        ctx.textAlign = "center";
        ctx.fillText("No data to display", width / 2, height / 2);
        return;
      }

      const margin = { top: 22, right: 22, bottom: 34, left: 66 };
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const [minY, maxY] = extent(rows, active);
      const xFor = (i) => margin.left + (rows.length === 1 ? 0 : (i / (rows.length - 1)) * plotW);
      const yFor = (v) => margin.top + (1 - (v - minY) / (maxY - minY)) * plotH;

      ctx.strokeStyle = "#d7ddd9";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#66736d";
      ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 5; i += 1) {
        const y = margin.top + (i / 5) * plotH;
        const value = maxY - (i / 5) * (maxY - minY);
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(width - margin.right, y);
        ctx.stroke();
        ctx.fillText(formatPct(value, 2), margin.left - 8, y);
      }

      const zeroY = yFor(0);
      if (zeroY >= margin.top && zeroY <= margin.top + plotH) {
        ctx.strokeStyle = "#17211d";
        ctx.globalAlpha = 0.45;
        ctx.beginPath();
        ctx.moveTo(margin.left, zeroY);
        ctx.lineTo(width - margin.right, zeroY);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      for (const factor of active) {
        for (const item of chartSeriesKeys(factor)) {
          ctx.strokeStyle = colors[factor] || "#17211d";
          ctx.lineWidth = item.width;
          ctx.globalAlpha = item.alpha;
          ctx.setLineDash(item.dash);
          ctx.beginPath();
          let started = false;
          rows.forEach((row, i) => {
            const value = row[item.key];
            if (value === null || value === undefined || Number.isNaN(value)) {
              started = false;
              return;
            }
            const x = xFor(i);
            const y = yFor(value);
            if (!started) {
              ctx.moveTo(x, y);
              started = true;
            } else {
              ctx.lineTo(x, y);
            }
          });
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
        }
      }

      drawAxisDates(rows, margin, plotW, height);
      if (state.hoverIndex !== null && rows[state.hoverIndex]) {
        const x = xFor(state.hoverIndex);
        ctx.strokeStyle = "#17211d";
        ctx.globalAlpha = 0.55;
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, margin.top + plotH);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    function drawAxisDates(rows, margin, plotW, height) {
      const ticks = Math.min(6, rows.length);
      ctx.fillStyle = "#66736d";
      ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i = 0; i < ticks; i += 1) {
        const index = Math.round((i / Math.max(1, ticks - 1)) * (rows.length - 1));
        const x = margin.left + (index / Math.max(1, rows.length - 1)) * plotW;
        ctx.fillText(rows[index].date.slice(0, 7), x, height - 24);
      }
    }

    function appendStatsRow(label, color, values) {
      if (!values.length) return;
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / Math.max(1, values.length - 1);
      const std = Math.sqrt(variance);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const pos = values.filter((v) => v > 0).length / values.length;
      const latest = values[values.length - 1];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="swatch" style="display:inline-block;vertical-align:-1px;background:${color}"></span> ${label}</td>
        <td>${formatPct(mean)}</td>
        <td>${formatPct(std)}</td>
        <td>${formatPct(min)}</td>
        <td>${formatPct(max)}</td>
        <td>${(pos * 100).toFixed(1)}%</td>
        <td>${formatPct(latest)}</td>
      `;
      statsBody.appendChild(tr);
    }

    function updateStats(rows, active) {
      statsBody.innerHTML = "";
      for (const factor of active) {
        for (const item of chartSeriesKeys(factor)) {
          const values = rows
            .map((row) => row[item.key])
            .filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
          appendStatsRow(item.label, colors[factor], values);
        }
      }
    }

    function handleHover(event) {
      const rows = displayRecords();
      if (!rows.length) return;
      const rect = canvas.getBoundingClientRect();
      const margin = { left: 66, right: 22 };
      const x = event.clientX - rect.left;
      const plotW = rect.width - margin.left - margin.right;
      const raw = (x - margin.left) / Math.max(1, plotW);
      const index = Math.max(0, Math.min(rows.length - 1, Math.round(raw * (rows.length - 1))));
      state.hoverIndex = index;
      const row = rows[index];
      const lines = [];
      for (const factor of state.active) {
        for (const item of chartSeriesKeys(factor)) {
          lines.push(
            `<div class="tooltip-line"><span>${item.label}</span><strong>${formatPct(row[item.key])}</strong></div>`
          );
        }
      }
      const linesHtml = lines.join("");
      tooltip.innerHTML = `<div class="tooltip-date">${row.date}</div>${linesHtml}`;
      tooltip.style.display = "block";
      tooltip.style.left = `${Math.min(rect.width - 330, Math.max(8, event.clientX - rect.left + 12))}px`;
      tooltip.style.top = `${Math.max(8, event.clientY - rect.top + 12)}px`;
      draw();
    }

    function downloadCsv() {
      const rows = displayRecords();
      const columns = ["date"];
      for (const factor of factorColumns) {
        for (const item of chartSeriesKeys(factor)) columns.push(item.key);
      }
      const uniqueColumns = [...new Set(columns)];
      const csv = [
        uniqueColumns.join(","),
        ...rows.map((row) => uniqueColumns.map((column) => row[column] ?? "").join(","))
      ].join("\\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const mode = cumulativeInput.checked ? "cumulative" : "daily";
      const seriesMode = currentSeriesMode();
      a.download = `ff5_${seriesMode}_${mode}_${startInput.value}_${endInput.value}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    initControls();
    draw();
  </script>
</body>
</html>
"""
