from typing import Any

import numpy as np
import torch
import xarray as xr

from starter_kit.baselines.mlp import _normalisation_mean, _normalisation_std
from starter_kit.baselines.sundquist import SundquistNetwork
from starter_kit.layers import InputNormalisation
from starter_kit.model import BaseModel


class PixelWiseClassifier(torch.nn.Module):
    def __init__(
        self, in_channels=3, levels=7, aux_channels=0, hidden=64, num_classes=3
    ):
        super().__init__()
        self.feat_dim = in_channels * levels + aux_channels
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(self.feat_dim, hidden, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L*in_channels + aux_channels, H, W)
        x = x.view(x.size(0), self.feat_dim, x.size(-2), x.size(-1))
        logits = self.net(x)  # (B, num_classes, H, W)
        return logits


class ClassificationNetwork(torch.nn.Module):
    """
    Plots showed limited ability of the current models to predict zeros and ones.
    Perform a classification task for each pixel to inflate these:
    three classes 0, 1 or in-between.
    First run the model through
    """

    def __init__(
        self, H=64, W=64, L=7, feature_index=[0, 1, 2], aux_index=[0, 1]
    ) -> None:
        super().__init__()

        self.L = L
        self.H = H
        self.W = W

        # Limit to inputs expected to have the most impact, to limit over-fitting
        self.classification_features_index = feature_index
        # 0, "temperature"
        # 1, "specific_humidity"
        # 2, "geopot"

        self.classification_aux_index = aux_index
        # 0,  # land sea mask
        # 1,  # geopotential

        self.classifier = ()

        self.normalization = InputNormalisation(
            mean=_normalisation_mean, std=_normalisation_std
        )

        self.pixel_classifier = PixelWiseClassifier(
            in_channels=len(self.classification_features_index),
            levels=self.L,
            aux_channels=len(self.classification_aux_index),
            hidden=16,
            num_classes=3,
        )

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor:

        # We collapse all levels into the channel dimension
        selected_inputs = input_level[:, self.classification_features_index, :, :, :]
        flattened_input_level = selected_inputs.reshape(
            selected_inputs.shape[0], -1, *selected_inputs.shape[-2:]
        )
        # output shape: (B, L*in_channels, H, W)

        # Move the feature dimension to the end for normalisation and MLP
        mlp_input = flattened_input_level.movedim(1, -1)
        # output shape: (B, H, W, L*in_channels)

        mlp_input = self.normalization(mlp_input)

        # Reshape to what PixelWiseClassifier expects
        mlp_input = torch.cat(
            [
                mlp_input.movedim(-1, 1),
                input_auxiliary[:, self.classification_aux_index, :, :],
            ],
            dim=1,
        )
        # output shape: (B, L*in_channels + aux_channels, H, W)

        # Classify:
        predicted_class = self.pixel_classifier(mlp_input)
        return predicted_class


class SundquistClassifier(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.sundquist = SundquistNetwork()
        self.classifier = ClassificationNetwork()

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor:

        # Classify:
        predicted_class = self.classifier(input_level, input_auxiliary)

        # Sundquist prediction:
        sundquist_output = self.sundquist.forward(input_level, input_auxiliary)

        return (
            sundquist_output * predicted_class[predicted_class == 1]
            + predicted_class[predicted_class != 1]
        )  # Keep only the "in-between" class


def estimate_cross_entropy(
    predictions: xr.DataArray,
    targets: xr.DataArray,
    tol: float = 1e-8,
) -> xr.DataArray:
    """
    Compute cross entropy loss between predictions and targets.

    Parameters
    ----------
    predictions: xr.DataArray
        Predicted values.
    targets: xr.DataArray
        Target observations.

    Returns:
        Cross entropy loss as xarray.DataArray
    """
    assert tol >= 0, "Tolerance must be positive"

    targets[targets <= 0 + tol] = 0
    targets[0 + tol < targets < 1 - tol] = 1
    targets[targets >= 1 - tol] = 2

    return _xarray_cross_entropy(predictions, targets)


def _xarray_cross_entropy(
    predictions: xr.DataArray, targets: xr.DataArray
) -> xr.DataArray:
    # Convert to PyTorch tensors
    pred_tensor = torch.from_numpy(predictions.values)
    target_tensor = torch.from_numpy(targets.values)

    # Compute cross entropy loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(pred_tensor, target_tensor)

    # Convert back to xarray.DataArray
    return xr.DataArray(loss.item(), dims=predictions.dims, coords=predictions.coords)


class ClassModel(BaseModel):
    def estimate_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        r"""
        Compute the primary training loss and prediction output.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing ``input_level``,
            ``input_auxiliary``, and ``target`` tensors.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``loss`` and ``prediction``.
            ``loss`` is the mean absolute error and ``prediction`` is the
            model output clamped to ``[0, 1]``.
        """
        prediction = self.network(
            input_level=batch["input_level"], input_auxiliary=batch["input_auxiliary"]
        )
        loss = estimate_cross_entropy(prediction, batch["target"])

        return {"loss": loss, "prediction": prediction}

    def estimate_auxiliary_loss(
        self, batch: dict[str, torch.Tensor], outputs: dict[str, Any]
    ) -> dict[str, Any]:
        r"""
        Compute auxiliary regression and classification metrics.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing the ground-truth ``target`` tensor.
        outputs : Dict[str, Any]
            Model outputs from ``estimate_loss`` containing ``prediction``.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``mse`` and ``accuracy``.
            ``mse`` is the mean squared error and ``accuracy`` is the
            thresholded classification accuracy at 0.5.
        """
        mse = (outputs["prediction"] - batch["target"]).pow(2)
        mse = (mse * self.lat_weights).mean()
        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = (prediction_bool == target_bool).float()
        accuracy = (accuracy * self.lat_weights).mean()
        return {"mse": mse, "accuracy": accuracy}
