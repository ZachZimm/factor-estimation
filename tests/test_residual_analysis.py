from __future__ import annotations

import re

import pandas as pd

from ff5_predictor.residual_analysis import (
    build_residual_panel,
    run_residual_analysis_from_frames,
)


TARGETS = ["Mkt-RF", "SMB"]


def _config() -> dict:
    return {
        "prediction": {"target_columns": TARGETS},
        "residual_analysis": {
            "autocorrelation_lags": [1, 2],
            "market_lags": [-1, 0, 1],
            "high_vol_quantile": 0.8,
            "low_vol_quantile": 0.2,
        },
    }


def _predictions() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=80)
    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "date": date.date().isoformat(),
                "target_date": date.date().isoformat(),
                "cutoff_date": (date - pd.Timedelta(days=1)).date().isoformat(),
                "gap_day": 1,
                "release_gap_size": 1,
                "model_type": "elasticnet",
                "pred_Mkt-RF": 0.001 * i,
                "actual_Mkt-RF": 0.001 * i + 0.0001,
                "pred_SMB": -0.0005 * i,
                "actual_SMB": -0.0005 * i - 0.0002,
            }
        )
    return pd.DataFrame(rows)


def _market() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=80)
    close = [100.0 + i for i in range(len(dates))]
    vix = [15.0 + (i % 10) for i in range(len(dates))]
    return pd.DataFrame(
        {
            "SPY_open": close,
            "SPY_high": close,
            "SPY_low": close,
            "SPY_close": close,
            "SPY_volume": [1] * len(dates),
            "^VIX_open": vix,
            "^VIX_high": vix,
            "^VIX_low": vix,
            "^VIX_close": vix,
            "^VIX_volume": [1] * len(dates),
            "HYG_open": close,
            "HYG_high": close,
            "HYG_low": close,
            "HYG_close": close,
            "HYG_volume": [1] * len(dates),
            "LQD_open": [100.0 + i * 0.5 for i in range(len(dates))],
            "LQD_high": [100.0 + i * 0.5 for i in range(len(dates))],
            "LQD_low": [100.0 + i * 0.5 for i in range(len(dates))],
            "LQD_close": [100.0 + i * 0.5 for i in range(len(dates))],
            "LQD_volume": [1] * len(dates),
        },
        index=dates,
    )


def test_build_residual_panel_sign_conventions() -> None:
    panel = build_residual_panel(_predictions().head(1), TARGETS, _config())

    first = panel.loc[panel["target"] == "Mkt-RF"].iloc[0]
    assert abs(float(first["official_minus_model_implied"]) - 0.0001) < 1e-12
    assert abs(float(first["model_implied_minus_official"]) + 0.0001) < 1e-12


def test_residual_analysis_writes_tables_and_figures(tmp_path) -> None:
    result = run_residual_analysis_from_frames(
        _predictions(),
        _market(),
        _config(),
        output_dir=tmp_path / "analysis",
        model_type="elasticnet",
        release_gap_size=1,
    )

    assert not result.residual_panel.empty
    assert not result.summary.empty
    assert (result.run_dir / "tables" / "residual_panel.csv").exists()
    assert (result.run_dir / "tables" / "residual_cross_correlation.csv").exists()
    assert (result.run_dir / "tables" / "residual_market_lead_lag_correlation.csv").exists()
    assert (result.run_dir / "tables" / "residual_regime_summary.csv").exists()
    assert (result.run_dir / "tables" / "residual_market_regression.csv").exists()
    assert (result.run_dir / "figures" / "elasticnet_residual_timeseries.svg").exists()
    assert (result.run_dir / "figures" / "elasticnet_residual_cross_correlation.svg").exists()
    overlay = result.run_dir / "figures" / "elasticnet_residual_market_overlay.svg"
    assert overlay.exists()
    overlay_text = overlay.read_text(encoding="utf-8")
    assert "SPY cumulative return" in overlay_text
    assert "mean abs residual" in overlay_text
    path_points = [
        (float(match.group(1)), float(match.group(2)))
        for match in re.finditer(r"[ML](-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)", overlay_text)
    ]
    assert path_points
    assert max(x for x, _ in path_points) <= 1120
    assert min(x for x, _ in path_points) >= 0
    assert result.metadata["residual_sign_convention"]["official_minus_model_implied"] == "actual - pred"
