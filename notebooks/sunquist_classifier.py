from typing import Any

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
        self.feature_index = feature_index
        self.aux_index = aux_index

        # Build normalization only for the selected feature channels.
        level_mean = torch.tensor(_normalisation_mean[:-2]).reshape(-1, self.L)
        level_std = torch.tensor(_normalisation_std[:-2]).reshape(-1, self.L)
        selected_level_mean = level_mean[self.feature_index].reshape(-1)
        selected_level_std = level_std[self.feature_index].reshape(-1)

        aux_mean = torch.tensor(_normalisation_mean[-2:])[self.aux_index]
        aux_std = torch.tensor(_normalisation_std[-2:])[self.aux_index]

        self.normalization = InputNormalisation(
            mean=torch.cat([selected_level_mean, aux_mean]),
            std=torch.cat([selected_level_std, aux_std]),
        )

        self.pixel_classifier = PixelWiseClassifier(
            in_channels=len(self.feature_index),
            levels=self.L,
            aux_channels=len(self.aux_index),
            hidden=16,
            num_classes=3,
        )

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor:

        # Select only the requested feature channels and collapse levels
        selected_inputs = input_level[:, self.feature_index, :, :, :]
        flattened_input_levels = selected_inputs.reshape(
            selected_inputs.shape[0], -1, *selected_inputs.shape[-2:]
        )
        selected_auxiliary = input_auxiliary[:, self.aux_index, :, :]

        # Concatenate the selected level and auxiliary fields
        mlp_input = torch.cat([flattened_input_levels, selected_auxiliary], dim=1)
        # output shape: (B, len(feature_index)*L + len(aux_index), H, W)

        # Move the feature dimension to the end for normalisation and MLP
        mlp_input = mlp_input.movedim(1, -1)
        # output shape: (B, H, W, selected_feature_dim)

        mlp_input = self.normalization(mlp_input)

        # Reshape to what PixelWiseClassifier expects
        mlp_input = mlp_input.movedim(-1, 1)
        # output shape: (B, selected_feature_dim, H, W)

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
        predicted_label = predicted_class.argmax(dim=1, keepdim=True)

        # Sundquist prediction:
        sundquist_output = self.sundquist.forward(input_level, input_auxiliary)

        # Build a combined prediction:
        # - class 0 => exact zero
        # - class 1 => use Sundquist output
        # - class 2 => exact one
        output = torch.zeros_like(sundquist_output)
        output = torch.where(predicted_label == 2, torch.ones_like(output), output)
        output = torch.where(predicted_label == 1, sundquist_output, output)
        return output


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

    target_copy = targets.copy()
    target_copy[target_copy <= 0 + tol] = 0
    target_copy[0 + tol < target_copy < 1 - tol] = 1
    target_copy[target_copy >= 1 - tol] = 2

    return _xarray_cross_entropy(predictions, target_copy)


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
