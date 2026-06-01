from __future__ import annotations


VISIBLE_MODELS = [
    "rolling_mean",
    "ewma",
    "ridge",
    "per_target_pls_ridge",
    "elasticnet",
    "per_factor_elasticnet",
    "tft",
]

AVAILABLE_MODELS = VISIBLE_MODELS
TORCH_MODEL_NAMES = {"tft", "torch_tft"}
TABULAR_MODEL_NAMES = {"ridge", "per_target_pls_ridge", "elasticnet", "per_factor_elasticnet"}
