from typing import Any

import torch

from starter_kit.baselines.mlp import _normalisation_mean, _normalisation_std
from starter_kit.baselines.sundquist import SundquistNetwork
from starter_kit.baselines.utils import (
    approximate_surface_pressure,
    estimate_relative_humidity,
)
from starter_kit.layers import InputNormalisation, PerPixelLinear

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
        flattened_input_level = input_level.reshape(
            input_level.shape[0], -1, *input_level.shape[-2:]
        )

        # Concatenate the level and auxiliary fields
        mlp_input = torch.cat([flattened_input_level, input_auxiliary], dim=1)

        # Move the feature dimension to the end for normalisation and MLP
        mlp_input = mlp_input.movedim(1, -1)

        mlp_input = self.normalization(mlp_input)

        # Apply the correction head
        correction = self.correction_head(mlp_input)  # (B, H * W)

        # Move the channel dimension to the expected position
        correction = correction.movedim(-1, 1)

        # Reshape correction to match the spatial dimensions
        correction = correction.view(sundquist_output.size(0), 1, H, W)  # (B, 1, H, W)

        return self.sigmoid(sundquist_output + correction)


class SundquistSimpleVerticalNetwork(torch.nn.Module):
    """
    Replace the random overlap assumption with some learnable pixel-wise weights.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.sundquist = SundquistNetwork()

        self.L = 7
        self.H = 64
        self.W = 64
        self.overlap_weights = torch.nn.Parameter(torch.ones(self.L, self.H, self.W))

        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor:

        sundquist_output = self.sundquist.forward_vertical(
            input_level, input_auxiliary
        )  # (B, PL, H, W)

        # Apply pixel-wise overlap correction
        overlap = (sundquist_output * self.overlap_weights.unsqueeze(0)).sum(
            dim=1, keepdim=True
        )
        correction = self.sigmoid(overlap)  # (B, 1, H, W)
        return correction


class SundquistResidualVerticalNetwork(torch.nn.Module):
    """
    Replace the random overlap assumption with some learnable pixel-wise weighting scheme,
    with a fully-connected layer.
    """

    def __init__(self, column_hidden_size=4) -> None:
        super().__init__()
        self.sundquist = SundquistNetwork()
        self.L = 7
        self.H = 64
        self.W = 64
        self.normalization = InputNormalisation(
            mean=_normalisation_mean, std=_normalisation_std
        )

        self.pixel_column = torch.nn.Sequential(
            PerPixelLinear(
                in_features=self.L, out_features=column_hidden_size, H=self.H, W=self.W
            ),
            torch.nn.ReLU(),
            PerPixelLinear(
                in_features=column_hidden_size, out_features=self.L, H=self.H, W=self.W
            ),
            torch.nn.Sigmoid(),
        )

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor:

        # Sundquist per-level estimate
        sundquist_output_level = self.sundquist.forward_vertical(
            input_level, input_auxiliary
        )  # (B, PL, H, W)

        # Residual estimate:
        # We collapse all levels into the channel dimension
        flattened_input_level = input_level.reshape(
            input_level.shape[0], -1, *input_level.shape[-2:]
        )

        # Concatenate the level and auxiliary fields
        mlp_input = torch.cat([flattened_input_level, input_auxiliary], dim=1)

        # Move the feature dimension to the end for normalisation and MLP
        mlp_input = flattened_input_level.movedim(1, -1)

        mlp_input = self.normalization(mlp_input)  # (B, L, H, W)

        # Apply the correction head
        correction_per_level = self.pixel_column(mlp_input)

        # Move the channel dimension to the expected position
        correction_per_level = correction_per_level.movedim(-1, 1)  # (B, L, H * W)

        # Reshape correction to match the spatial dimensions
        correction_per_level = correction_per_level.view(
            sundquist_output_level.size(0), 1, self.H, self.W
        )  # (B, 1, H, W)

        # Combine
        combined_per_level = (
            sundquist_output_level + correction_per_level
        )  # (B, L, H, W)

        # Combine
        return total_cloud_cover
