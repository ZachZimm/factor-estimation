from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ff5_predictor.torch_models import make_torch_model


@pytest.mark.parametrize("model_type", ["tft", "tcn", "ft_transformer"])
def test_torch_model_forward_shape(model_type: str) -> None:
    cfg = {
        "torch": {"hidden_size": 8, "dropout": 0.0},
        "models": {
            "tft": {"hidden_size": 8, "n_heads": 2, "lstm_layers": 1, "dropout": 0.0},
            "tcn": {"channels": [4, 4], "kernel_size": 3, "dropout": 0.0},
            "ft_transformer": {
                "d_model": 8,
                "n_heads": 2,
                "num_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
            },
        },
    }
    model = make_torch_model(model_type, lookback_rows=5, n_features=3, n_targets=2, config=cfg)
    out = model(torch.randn(4, 5, 3))

    assert out.shape == (4, 2)
    assert not torch.isnan(out).any()
