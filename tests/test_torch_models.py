from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ff5_predictor.torch_models import make_torch_model


def test_tft_forward_shape() -> None:
    cfg = {
        "torch": {"hidden_size": 8, "dropout": 0.0},
        "models": {
            "tft": {"hidden_size": 8, "n_heads": 2, "lstm_layers": 1, "dropout": 0.0},
        },
    }
    model = make_torch_model("tft", lookback_rows=5, n_features=3, n_targets=2, config=cfg)
    out = model(torch.randn(4, 5, 3))

    assert out.shape == (4, 2)
    assert not torch.isnan(out).any()
