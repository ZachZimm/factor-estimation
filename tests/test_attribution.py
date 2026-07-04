from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from ff5_predictor.attribution import explain_linear_predictions, explain_ridge_predictions, infer_feature_group
from ff5_predictor.nowcast_models import FittedNowcastModel


def _fitted() -> tuple[FittedNowcastModel, pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(
        {
            "proxy_size_ijr_spy": [0.01, -0.02, 0.03],
            "proxy_value_iwn_iwo": [0.02, 0.01, -0.01],
            "SPY_vol_21d": [0.10, 0.20, 0.15],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )
    targets = pd.DataFrame(
        {
            "Mkt-RF": [0.01, -0.01, 0.02],
            "SMB": [0.001, 0.002, -0.001],
        },
        index=features.index,
    )
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=0.1).fit(scaler.transform(features), targets)
    fitted = FittedNowcastModel(
        model_type="ridge",
        model=model,
        scaler=scaler,
        feature_columns=list(features.columns),
        target_columns=list(targets.columns),
        metadata={
            "alpha": 0.1,
            "scale_features": True,
            "train_start_date": "2024-01-01",
            "train_end_date": "2024-01-03",
            "n_train_rows": 3,
        },
    )
    pred = pd.DataFrame({"date": features.index.date.astype(str)})
    predicted = fitted.predict_frame(features)
    for idx, target in enumerate(fitted.target_columns):
        pred[f"pred_{target}"] = predicted[:, idx]
    return fitted, features, pred


def test_ridge_attribution_reconstructs_predictions_and_sorts_top_features() -> None:
    fitted, features, predictions = _fitted()

    result = explain_ridge_predictions(fitted, features, predictions, top_n=2)

    assert len(result.coefficient_table) == len(fitted.feature_columns) * len(fitted.target_columns)
    assert set(result.top_contributions["rank"]) == {1, 2}
    for (_, _), group in result.top_contributions.groupby(["date", "target"]):
        assert group["abs_contribution"].tolist() == sorted(group["abs_contribution"], reverse=True)
    assert not result.group_summary.empty
    assert not result.contribution_wide.empty


def test_elasticnet_attribution_reconstructs_predictions() -> None:
    fitted, features, _ = _fitted()
    targets = pd.DataFrame(
        {
            "Mkt-RF": [0.01, -0.01, 0.02],
            "SMB": [0.001, 0.002, -0.001],
        },
        index=features.index,
    )
    scaler = StandardScaler().fit(features)
    model = MultiTaskElasticNet(alpha=0.0001, l1_ratio=0.05, max_iter=10000).fit(scaler.transform(features), targets)
    fitted.model_type = "elasticnet"
    fitted.model = model
    fitted.scaler = scaler
    fitted.metadata["alpha"] = 0.0001
    predictions = pd.DataFrame({"date": features.index.date.astype(str)})
    predicted = fitted.predict_frame(features)
    for idx, target in enumerate(fitted.target_columns):
        predictions[f"pred_{target}"] = predicted[:, idx]

    result = explain_linear_predictions(fitted, features, predictions, top_n=2)

    assert result.metadata["model_type"] == "elasticnet"
    assert len(result.coefficient_table) == len(fitted.feature_columns) * len(fitted.target_columns)
    assert not result.top_contributions.empty


def test_feature_group_inference() -> None:
    assert infer_feature_group("proxy_size_ijr_spy") == "proxy_size"
    assert infer_feature_group("proxy_value_iwn_iwo") == "proxy_value"
    assert infer_feature_group("proxy_global_eem_spy") == "proxy_global"
    assert infer_feature_group("SPY_vol_21d") == "rolling_volatility"


def test_attribution_rejects_unsupported_model_type() -> None:
    fitted, features, predictions = _fitted()
    fitted.model_type = "tft"

    with pytest.raises(ValueError, match="Linear attribution"):
        explain_linear_predictions(fitted, features, predictions, top_n=2)


def test_attribution_detects_reconstruction_errors() -> None:
    fitted, features, predictions = _fitted()
    predictions.loc[0, "pred_Mkt-RF"] += 1.0

    with pytest.raises(ValueError, match="reconstruction failed"):
        explain_ridge_predictions(fitted, features, predictions, top_n=2)
