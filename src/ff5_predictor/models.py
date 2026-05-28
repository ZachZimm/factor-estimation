from __future__ import annotations

from typing import Any

from sklearn.preprocessing import StandardScaler

from ff5_predictor.tabular_models import make_tabular_model


def make_model(model_type: str, config: dict[str, Any]):
    return make_tabular_model(model_type, config)


def make_scaler(enabled: bool):
    return StandardScaler() if enabled else None
