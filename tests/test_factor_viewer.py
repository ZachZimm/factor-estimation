from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ff5_predictor.factor_viewer import (
    prepare_predictions_for_viewer,
    resolve_predictions_path,
    write_factor_viewer_html,
)

TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _ff5_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=5)
    return pd.DataFrame({col: [0.001 * (i + 1) for i in range(5)] for col in TARGETS}, index=dates)


def _backtest_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-04",
                "target_date": "2024-01-04",
                "model_type": "ridge",
                "gap_day": 1,
                "pred_Mkt-RF": 0.002,
                "actual_Mkt-RF": 0.003,
                "pred_SMB": 0.001,
                "actual_SMB": 0.002,
            },
            {
                "date": "2024-01-04",
                "target_date": "2024-01-04",
                "model_type": "ewma",
                "gap_day": 1,
                "pred_Mkt-RF": 0.0015,
                "actual_Mkt-RF": 0.003,
                "pred_SMB": 0.0012,
                "actual_SMB": 0.002,
            },
            {
                "date": "2024-01-05",
                "target_date": "2024-01-05",
                "model_type": "ridge",
                "gap_day": 2,
                "pred_Mkt-RF": 0.0025,
                "actual_Mkt-RF": 0.004,
                "pred_SMB": 0.0018,
                "actual_SMB": 0.0025,
            },
        ]
    )


def test_write_factor_viewer_without_predictions(tmp_path: Path) -> None:
    output = tmp_path / "viewer.html"
    write_factor_viewer_html(_ff5_frame(), output)

    html = output.read_text(encoding="utf-8")
    assert "__FF5_PAYLOAD__" not in html
    assert '"comparison":null' in html.replace(" ", "")
    assert "Factor Returns Over Time" in html


def test_write_factor_viewer_with_comparison_payload(tmp_path: Path) -> None:
    output = tmp_path / "comparison.html"
    predictions = _backtest_predictions()
    predictions["date"] = pd.to_datetime(predictions["date"])

    write_factor_viewer_html(
        _ff5_frame(),
        output,
        predictions_df=predictions,
        target_columns=TARGETS,
        default_model="ewma",
        default_gap_day=2,
        run_label="gap_test.csv",
    )

    html = output.read_text(encoding="utf-8")
    payload_start = html.index('const payload = ') + len("const payload = ")
    payload_end = html.index(";", payload_start)
    payload = json.loads(html[payload_start:payload_end])

    assert payload["comparison"]["enabled"] is True
    assert payload["comparison"]["defaultModel"] == "ewma"
    assert payload["comparison"]["defaultGapDay"] == 2
    assert payload["comparison"]["gapDays"] == [1, 2]
    assert set(payload["comparison"]["models"]) == {"ridge", "ewma"}
    assert len(payload["comparison"]["predictionRows"]) == 3
    assert payload["comparison"]["defaultSeriesMode"] == "both"
    assert "comparisonControls" in html
    assert "seriesMode" in html


def test_prepare_predictions_for_viewer_requires_rows() -> None:
    with pytest.raises(ValueError, match="empty"):
        prepare_predictions_for_viewer(pd.DataFrame(), TARGETS)


def test_resolve_predictions_path_from_run_dir(tmp_path: Path) -> None:
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir(parents=True)
    preferred = predictions_dir / "release_gap_predictions.csv"
    preferred.write_text("date,model_type,pred_Mkt-RF\n2024-01-02,ridge,0.001\n", encoding="utf-8")
    (predictions_dir / "other.csv").write_text("date\n2024-01-02\n", encoding="utf-8")

    resolved = resolve_predictions_path(run_dir=tmp_path)
    assert resolved == preferred


def test_resolve_predictions_path_missing_run_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_predictions_path(run_dir=tmp_path / "missing")
