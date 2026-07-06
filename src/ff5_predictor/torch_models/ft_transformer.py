from __future__ import annotations

import torch
from torch import nn


class SequenceFTTransformerRegressor(nn.Module):
    """FT-Transformer-style tabular model with simple temporal summaries.

    The model receives a lookback window but builds one token per feature using
    last, mean, and standard-deviation summaries across the window. This keeps
    attention over features rather than over every time-feature pair.
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.value_projection = nn.Linear(3, d_model)
        self.feature_embedding = nn.Parameter(torch.zeros(1, n_features, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_targets)
        nn.init.normal_(self.feature_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :]
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        tokens = torch.stack([last, mean, std], dim=-1)
        encoded = self.value_projection(tokens) + self.feature_embedding
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, encoded], dim=1))
        return self.head(self.norm(encoded[:, 0, :]))
