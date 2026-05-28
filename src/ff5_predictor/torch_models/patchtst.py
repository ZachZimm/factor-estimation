from __future__ import annotations

import torch
from torch import nn


class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        lookback_rows: int,
        n_features: int,
        n_targets: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        n_patches = 1 + max(0, (lookback_rows - patch_len) // stride)
        if n_patches < 1:
            raise ValueError("patch_len must be <= lookback_rows")
        self.proj = nn.Linear(patch_len * n_features, d_model)
        self.positional = nn.Parameter(torch.zeros(1, n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).flatten(start_dim=2)
        encoded = self.proj(patches) + self.positional[:, : patches.shape[1], :]
        encoded = self.encoder(encoded)
        return self.head(encoded.mean(dim=1))
