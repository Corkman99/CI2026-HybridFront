from typing import Any

import torch

from starter_kit.baselines.mlp import _normalisation_mean, _normalisation_std
from starter_kit.baselines.sundquist import SundquistNetwork
from starter_kit.baselines.utils import (
    approximate_surface_pressure,
    estimate_relative_humidity,
)
from starter_kit.layers import InputNormalisation
from starter_kit.model import BaseModel

H = 64
W = 64


class SundquistPlusNetwork(torch.nn.Module):

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.sundquist = SundquistNetwork()

        # Example extra head on top of Sundquist output
        self.correction_head = torch.nn.Sequential(
            torch.nn.Linear(H * W, H * W),
            torch.torch.nn.sigmoid(),
        )

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor:
        # sunshape
        sundquist_output = self.sundquist(
            input_level, input_auxiliary
        )  # (B, PL=1, H, W)

        # We collapse all levels into the channel dimension
        flattened_output = sundquist_output.reshape(
            sundquist_output.shape[0], -1, *sundquist_output.shape[-2:]
        )

        # Move the feature dimension to the end for normalisation and MLP
        mlp_input = flattened_output.movedim(1, -1)

        # Apply the correction head
        correction = self.correction_head(mlp_input)  # (B, H * W)

        # Move the channel dimension to the expected position
        correction = correction.movedim(-1, 1)

        # Reshape correction to match the spatial dimensions
        correction = correction.view(sundquist_output.size(0), 1, H, W)  # (B, 1, H, W)

        return correction


class SundquistVerticalNetwork(torch.nn.Module):

    def __init__(
        self,
    ) -> None: ...

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor: ...
