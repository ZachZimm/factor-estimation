from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size] if self.chomp_size else x


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.proj(x)


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
        blocks: list[nn.Module] = []
        in_channels = n_features
        for idx, out_channels in enumerate(channels):
            blocks.append(
                ResidualTCNBlock(
                    in_channels=in_channels,
                    out_channels=int(out_channels),
                    kernel_size=kernel_size,
                    dilation=2**idx,
                    dropout=dropout,
                )
            )
            in_channels = int(out_channels)
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(in_channels, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        encoded = self.tcn(x)
        pooled = encoded.mean(dim=-1)
        return self.head(pooled)
