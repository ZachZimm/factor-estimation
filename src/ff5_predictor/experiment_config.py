from __future__ import annotations


VISIBLE_MODELS = [
    "rolling_mean",
    "ewma",
    "ridge",
    "per_target_pls_ridge",
    "elasticnet",
    "elasticnet_mom_override",
    "per_factor_elasticnet",
    "gradient_boosting",
    "tcn",
    "ft_transformer",
    "tft",
]

AVAILABLE_MODELS = VISIBLE_MODELS
TORCH_MODEL_NAMES = {"tft", "torch_tft", "tcn", "torch_tcn", "ft_transformer", "torch_ft_transformer"}
TABULAR_MODEL_NAMES = {
    "ridge",
    "per_target_pls_ridge",
    "elasticnet",
    "elasticnet_mom_override",
    "per_factor_elasticnet",
    "gradient_boosting",
}
