from __future__ import annotations

from ff5_predictor.experiment_config import AVAILABLE_MODELS, VISIBLE_MODELS


def test_only_current_nowcast_models_are_listed() -> None:
    assert "mlp_window" not in VISIBLE_MODELS
    assert "tcn" not in VISIBLE_MODELS
    assert "patchtst" not in VISIBLE_MODELS
    assert "mlp_window" not in AVAILABLE_MODELS
    assert "tcn" not in AVAILABLE_MODELS
    assert "patchtst" not in AVAILABLE_MODELS
    assert "tft" in VISIBLE_MODELS
