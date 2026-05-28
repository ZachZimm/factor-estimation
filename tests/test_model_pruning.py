from __future__ import annotations

from ff5_predictor.experiment_config import AVAILABLE_MODELS, HIDDEN_MODELS, VISIBLE_MODELS


def test_mlp_and_tcn_hidden_from_visible_models() -> None:
    assert "mlp_window" not in VISIBLE_MODELS
    assert "tcn" not in VISIBLE_MODELS
    assert "patchtst" not in VISIBLE_MODELS
    assert "mlp_window" in HIDDEN_MODELS
    assert "tcn" in HIDDEN_MODELS
    assert "patchtst" in HIDDEN_MODELS
    assert "mlp_window" in AVAILABLE_MODELS
    assert "tcn" in AVAILABLE_MODELS
    assert "patchtst" in AVAILABLE_MODELS
    assert "tft" in VISIBLE_MODELS
