from __future__ import annotations

from typing import Any

import pandas as pd

from ff5_predictor.evaluation import evaluate_prediction_groups


def add_regime_labels(
    predictions: pd.DataFrame,
    modeling_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    regimes = config.get("regimes", {})
    if not regimes.get("enabled", True) or predictions.empty:
        result = predictions.copy()
        result["regime_all"] = "all"
        return result

    result = predictions.copy()
    result["date"] = pd.to_datetime(result["date"])
    feature_frame = modeling_df.copy()
    feature_frame.index = pd.to_datetime(feature_frame.index)
    vol_col = str(regimes.get("volatility_source", "SPY_vol_21d"))
    ret_col = str(regimes.get("market_return_source", "SPY_ret_21d"))
    labels = pd.DataFrame(index=feature_frame.index)
    if vol_col in feature_frame.columns:
        vol = feature_frame[vol_col]
        low = vol.quantile(float(regimes.get("low_vol_quantile", 0.2)))
        high = vol.quantile(float(regimes.get("high_vol_quantile", 0.8)))
        labels["vol_regime"] = "normal_vol"
        labels.loc[vol <= low, "vol_regime"] = "low_vol"
        labels.loc[vol >= high, "vol_regime"] = "high_vol"
    else:
        labels["vol_regime"] = "all"
    if ret_col in feature_frame.columns:
        labels["market_regime"] = "market_down_21d"
        labels.loc[feature_frame[ret_col] >= 0, "market_regime"] = "market_up_21d"
    else:
        labels["market_regime"] = "all"
    return result.merge(labels, left_on="date", right_index=True, how="left")


def evaluate_by_regime(
    predictions: pd.DataFrame,
    target_columns: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    if predictions.empty:
        return {}
    output: dict[str, Any] = {
        "all": evaluate_prediction_groups(predictions, target_columns),
    }
    for regime_column in ["vol_regime", "market_regime"]:
        if regime_column not in predictions.columns:
            continue
        output[regime_column] = {}
        for regime_value, group in predictions.groupby(regime_column):
            output[regime_column][str(regime_value)] = evaluate_prediction_groups(group, target_columns)
    return output
