from __future__ import annotations

import torch
from torch import nn


class GatedResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.ff(x)
        return self.norm(x + self.gate(update) * update)


class VariableSelection(nn.Module):
    def __init__(self, n_features: int, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(n_features, n_features)
        self.feature_projection = nn.Linear(n_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        return self.feature_projection(x * weights)


class TFTStyleRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        hidden_size: int,
        n_heads: int,
        lstm_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.variable_selection = VariableSelection(n_features, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.grn = GatedResidualBlock(hidden_size, dropout)
        self.head = nn.Linear(hidden_size * 2, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected = self.variable_selection(x)
        encoded, _ = self.lstm(selected)
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        enriched = self.grn(attended)
        last = enriched[:, -1, :]
        context = enriched.mean(dim=1)
        return self.head(torch.cat([last, context], dim=-1))
