from __future__ import annotations

import torch
from torch import nn

from ff5_predictor.torch_models.common import activation_layer


class WindowMLP(nn.Module):
    def __init__(
        self,
        lookback_rows: int,
        n_features: int,
        n_targets: int,
        hidden_sizes: list[int],
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = lookback_rows * n_features
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_features, int(hidden_size)),
                    activation_layer(activation),
                    nn.Dropout(dropout),
                ]
            )
            in_features = int(hidden_size)
        layers.append(nn.Linear(in_features, n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))
