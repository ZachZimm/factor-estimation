from __future__ import annotations

import torch
from torch import nn


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x) + self.projection(x))


class TCNRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        channels: list[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = n_features
        for layer_idx, out_channels in enumerate(channels):
            blocks.append(
                ResidualTCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**layer_idx,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(in_channels * 2, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.network(x.transpose(1, 2))
        last = encoded[:, :, -1]
        pooled = encoded.mean(dim=2)
        return self.head(torch.cat([last, pooled], dim=1))
