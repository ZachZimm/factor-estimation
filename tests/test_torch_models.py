from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ff5_predictor.torch_models import make_torch_model


@pytest.mark.parametrize("model_type", ["mlp_window", "tcn", "patchtst", "tft"])
def test_torch_model_forward_shapes(model_type: str) -> None:
    cfg = {
        "torch": {"hidden_size": 8, "dropout": 0.0},
        "models": {
            "mlp_window": {"hidden_sizes": [16], "activation": "gelu", "dropout": 0.0},
            "tcn": {"channels": [8], "kernel_size": 3, "dropout": 0.0},
            "patchtst": {
                "patch_len": 2,
                "stride": 1,
                "d_model": 8,
                "n_heads": 2,
                "num_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
            },
            "tft": {"hidden_size": 8, "n_heads": 2, "lstm_layers": 1, "dropout": 0.0},
        },
    }
    model = make_torch_model(model_type, lookback_rows=5, n_features=3, n_targets=2, config=cfg)
    out = model(torch.randn(4, 5, 3))

    assert out.shape == (4, 2)
    assert not torch.isnan(out).any()
