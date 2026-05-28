from __future__ import annotations

import pandas as pd

from ff5_predictor.regime_evaluation import add_regime_labels, evaluate_by_regime


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def test_regime_labels_and_metrics_use_feature_columns() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    modeling = pd.DataFrame(
        {
            "SPY_vol_21d": range(10),
            "SPY_ret_21d": [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
        },
        index=dates,
    )
    preds = pd.DataFrame({"date": dates, "model_type": ["m"] * 10})
    for target in TARGETS:
        preds[f"pred_{target}"] = 0.0
        preds[f"actual_{target}"] = 0.0

    labeled = add_regime_labels(preds, modeling, {"regimes": {"enabled": True}})
    assert "high_vol" in set(labeled["vol_regime"])
    assert "low_vol" in set(labeled["vol_regime"])
    metrics = evaluate_by_regime(labeled, TARGETS, {})
    assert "vol_regime" in metrics
    assert "high_vol" in metrics["vol_regime"]
