from __future__ import annotations

import pandas as pd

from ff5_predictor.target_transform import apply_target_transform, reconstruct_predictions


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _df(rows: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = {"feature": range(rows)}
    for target in TARGETS:
        data[target] = [float(i + 1) for i in range(rows)]
    return pd.DataFrame(data, index=dates)


def test_daily_mode_is_noop() -> None:
    result = apply_target_transform(
        _df(),
        TARGETS,
        {"prediction": {"target_mode": "daily", "cumulative_horizon_rows": 1}},
    )
    assert result.target_columns == TARGETS
    assert result.modeling_df[TARGETS].equals(_df()[TARGETS])


def test_cumulative_5d_drops_incomplete_terminal_rows() -> None:
    result = apply_target_transform(
        _df(),
        TARGETS,
        {"prediction": {"target_mode": "cumulative", "cumulative_horizon_rows": 5}},
    )
    assert result.target_columns[0] == "Mkt-RF_fwd_5d"
    assert result.modeling_df.iloc[0]["Mkt-RF_fwd_5d"] == 15.0
    assert len(result.modeling_df) == 4


def test_residual_reconstruction_uses_prior_rolling_mean() -> None:
    result = apply_target_transform(
        _df(),
        TARGETS,
        {
            "prediction": {
                "target_mode": "residual",
                "cumulative_horizon_rows": 1,
                "residual_baseline": {"type": "rolling_mean", "window_rows": 3},
            }
        },
    )
    first_date = result.modeling_df.index[0]
    assert first_date == pd.Timestamp("2024-01-04")
    assert result.modeling_df.loc[first_date, "Mkt-RF_baseline"] == 2.0
    assert result.modeling_df.loc[first_date, "Mkt-RF_residual"] == 2.0
    preds = pd.DataFrame(
        {
            "date": [first_date],
            "model_type": ["test"],
            "pred_Mkt-RF_residual": [0.5],
            "actual_Mkt-RF_residual": [2.0],
        }
    )
    for target in TARGETS[1:]:
        preds[f"pred_{target}_residual"] = 0.5
        preds[f"actual_{target}_residual"] = 2.0
    reconstructed = reconstruct_predictions(preds, result, {})
    assert reconstructed.loc[0, "pred_Mkt-RF"] == 2.5
    assert reconstructed.loc[0, "actual_Mkt-RF"] == 4.0


def test_residual_5d_baseline_uses_historical_realized_5d_targets() -> None:
    result = apply_target_transform(
        _df(10),
        TARGETS,
        {
            "prediction": {
                "target_mode": "residual",
                "cumulative_horizon_rows": 5,
                "residual_baseline": {"type": "rolling_mean", "window_rows": 2},
            }
        },
    )
    first_date = result.modeling_df.index[0]
    assert first_date == pd.Timestamp("2024-01-03")
    assert result.modeling_df.loc[first_date, "Mkt-RF_baseline"] == (15.0 + 20.0) / 2
