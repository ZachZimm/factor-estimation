from __future__ import annotations


VISIBLE_MODELS = [
    "naive_previous",
    "rolling_mean",
    "rolling_median",
    "ewma",
    "ridge",
    "elasticnet",
    "lightgbm",
    "xgboost",
    "catboost",
    "tft",
]

HIDDEN_MODELS = ["patchtst", "mlp_window", "tcn"]
AVAILABLE_MODELS = VISIBLE_MODELS + HIDDEN_MODELS
TORCH_MODEL_NAMES = {"mlp_window", "tcn", "patchtst", "tft", "torch_mlp", "torch_tcn", "torch_patchtst", "torch_tft"}
TABULAR_MODEL_NAMES = {"ridge", "elasticnet", "lightgbm", "xgboost", "catboost"}
