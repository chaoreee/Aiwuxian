from __future__ import annotations

import math

import torch
from torch import nn


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int, levels: int = 8) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(levels, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", frequencies)
        self.input_dim = input_dim

    @property
    def output_dim(self) -> int:
        return self.input_dim * (1 + 2 * len(self.frequencies))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = x.unsqueeze(-1) * self.frequencies
        encoded = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(1)
        return torch.cat([x, encoded], dim=1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class PhysicalChannelField(nn.Module):
    """Coordinate-conditioned 3D neural decoder in angle-delay space."""

    def __init__(
        self,
        feature_dim: int = 11,
        width: int = 384,
        delay_bins: int = 16,
        fourier_levels: int = 8,
    ) -> None:
        super().__init__()
        if delay_bins != 16:
            raise ValueError("the current decoder is configured for 16 delay bins")
        self.delay_bins = delay_bins
        self.fourier = FourierFeatures(input_dim=4, levels=fourier_levels)
        encoded_dim = self.fourier.output_dim + feature_dim - 4
        self.encoder = nn.Sequential(
            nn.Linear(encoded_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            ResidualMLPBlock(width),
            ResidualMLPBlock(width),
            ResidualMLPBlock(width),
        )
        self.scale_head = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 1))
        self.coverage_head = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 1))
        self.seed = nn.Linear(width, 256 * 2 * 1 * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 160, 4, 2, 1),
            nn.GroupNorm(16, 160),
            nn.SiLU(),
            nn.ConvTranspose3d(160, 96, 4, 2, 1),
            nn.GroupNorm(12, 96),
            nn.SiLU(),
            nn.ConvTranspose3d(96, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 48, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(48, 16, 1),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = torch.cat([self.fourier(features[:, :4]), features[:, 4:]], dim=1)
        latent = self.encoder(encoded)
        seed = self.seed(latent).reshape(-1, 256, 2, 1, 2)
        shape = self.decoder(seed)
        return {
            "shape": shape,
            "log_rms": self.scale_head(latent).squeeze(1),
            "coverage_logit": self.coverage_head(latent).squeeze(1),
        }
