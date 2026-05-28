from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import MultiTaskElasticNet, Ridge


class PerTargetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.estimators_: list[Any] = []

    def fit(self, X, y):
        y_array = np.asarray(y)
        self.estimators_ = []
        for idx in range(y_array.shape[1]):
            estimator = clone(self.base_estimator)
            estimator.fit(X, y_array[:, idx])
            self.estimators_.append(estimator)
        return self

    def predict(self, X):
        return np.column_stack([estimator.predict(X) for estimator in self.estimators_])


def supports_multi_output(model_type: str) -> bool:
    return model_type in {"ridge", "elasticnet"}


def make_tabular_model(model_type: str, config: dict[str, Any]):
    model_cfg = config.get("models", {}).get(model_type, {})
    if model_type == "ridge":
        alpha = float(config.get("training", {}).get("ridge_alpha", 1.0))
        return Ridge(alpha=alpha)
    if model_type == "elasticnet":
        return MultiTaskElasticNet(
            alpha=float(model_cfg.get("alpha", 0.0001)),
            l1_ratio=float(model_cfg.get("l1_ratio", 0.2)),
            max_iter=int(model_cfg.get("max_iter", 10000)),
            tol=float(model_cfg.get("tol", 0.0001)),
        )
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install ff5-predictor[boosting] to use model_type='xgboost'") from exc
        base = XGBRegressor(
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            learning_rate=float(model_cfg.get("learning_rate", 0.03)),
            max_depth=int(model_cfg.get("max_depth", 3)),
            subsample=float(model_cfg.get("subsample", 0.8)),
            colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
            objective="reg:squarederror",
        )
        return PerTargetRegressor(base)
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Install ff5-predictor[boosting] to use model_type='lightgbm'") from exc
        base = LGBMRegressor(
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            learning_rate=float(model_cfg.get("learning_rate", 0.03)),
            max_depth=int(model_cfg.get("max_depth", -1)),
            num_leaves=int(model_cfg.get("num_leaves", 31)),
        )
        return PerTargetRegressor(base)
    if model_type == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("Install ff5-predictor[boosting] to use model_type='catboost'") from exc
        base = CatBoostRegressor(
            iterations=int(model_cfg.get("iterations", 300)),
            learning_rate=float(model_cfg.get("learning_rate", 0.03)),
            depth=int(model_cfg.get("depth", 4)),
            verbose=bool(model_cfg.get("verbose", False)),
            loss_function="RMSE",
        )
        return PerTargetRegressor(base)
    raise ValueError(f"Unsupported tabular model_type: {model_type}")
