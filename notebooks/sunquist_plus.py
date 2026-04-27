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
L = 7


class SundquistPlusNetwork(torch.nn.Module):

    def __init__(
        self,
        linear_bias: bool = False,
    ) -> None:
        super().__init__()

        self.sundquist = SundquistNetwork()

        self.normalization = InputNormalisation(
            mean=_normalisation_mean, std=_normalisation_std
        )
        self.correction_head = torch.nn.Linear(L * H * W, H * W, bias=linear_bias)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor:
        # Sunquist estimate
        sundquist_output = self.sundquist(
            input_level, input_auxiliary
        )  # (B, PL=1, H, W)

        # Residual estimate:
        # We collapse all levels into the channel dimension
        flattened_output = input_level.reshape(
            input_level.shape[0], -1, *input_level.shape[-2:]
        )

        # Move the feature dimension to the end for normalisation and MLP
        mlp_input = flattened_output.movedim(1, -1)

        mlp_input = self.normalization(mlp_input)

        # Apply the correction head
        correction = self.correction_head(mlp_input)  # (B, H * W)

        # Move the channel dimension to the expected position
        correction = correction.movedim(-1, 1)

        # Reshape correction to match the spatial dimensions
        correction = correction.view(sundquist_output.size(0), 1, H, W)  # (B, 1, H, W)

        return self.sigmoid(sundquist_output + correction)
