from __future__ import annotations

import pandas as pd

from ff5_predictor.rolling_train import naive_previous_value_baseline, rolling_predict


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _modeling_df(rows: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = {"feature": [float(value) for value in range(rows)]}
    for target in TARGETS:
        data[target] = [float(value) / 100 for value in range(rows)]
    return pd.DataFrame(data, index=dates)


def _config() -> dict:
    return {
        "training": {
            "train_window_days": 3,
            "min_train_rows": 3,
            "step_size": 1,
            "model_type": "ridge",
            "scale_features": True,
            "save_models": False,
        },
        "output": {"models_dir": "data/models"},
    }


def test_output_has_expected_columns() -> None:
    predictions = rolling_predict(_modeling_df(), ["feature"], TARGETS, _config())

    expected = {
        "date",
        "model_type",
        "train_start_date",
        "train_end_date",
        "n_train_rows",
    }
    expected.update({f"pred_{target}" for target in TARGETS})
    expected.update({f"actual_{target}" for target in TARGETS})
    assert expected.issubset(predictions.columns)


def test_naive_baseline_uses_previous_available_target() -> None:
    df = _modeling_df()
    predictions = naive_previous_value_baseline(df, TARGETS)

    first = predictions.iloc[0]
    assert pd.Timestamp(first["date"]) == pd.Timestamp("2024-01-02")
    for target in TARGETS:
        assert first[f"pred_{target}"] == df.iloc[0][target]
        assert first[f"actual_{target}"] == df.iloc[1][target]
