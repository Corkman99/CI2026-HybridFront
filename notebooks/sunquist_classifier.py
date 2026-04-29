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

    def __init__(self, input_dim=0, hidden_dim=0, n_layers=0) -> None:
        super().__init__()

        self.L = 7
        self.H = 64
        self.W = 64
        self.feature_index = [0, 1, 2]  # temperature, relative humidity, geopotential
        self.aux_index = [0, 1]  # land sea mask and geopotential only

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
    predictions: Any,
    targets: Any,
    tol: float = 1e-8,
) -> torch.Tensor:
    """
    Compute cross entropy loss between predictions and targets.

    Parameters
    ----------
    predictions : torch.Tensor or xr.DataArray
        Predicted logits with shape ``(B, num_classes, H, W)``.
    targets : torch.Tensor or xr.DataArray
        Target values in ``[0, 1]`` that will be mapped to class labels.
    tol : float, optional
        Tolerance used to bin target values into classes 0, 1, or 2.

    Returns
    -------
    torch.Tensor
        Scalar cross entropy loss.
    """
    assert tol >= 0, "Tolerance must be positive"
    targets_ = targets.clone().detach()

    if isinstance(predictions, xr.DataArray):
        predictions = torch.as_tensor(predictions.values)
    if isinstance(targets_, xr.DataArray):
        targets_ = torch.as_tensor(targets_.values)

    if not isinstance(predictions, torch.Tensor):
        raise TypeError("predictions must be a torch.Tensor or xarray.DataArray")
    if not isinstance(targets_, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor or xarray.DataArray")

    targets_ = targets_.to(predictions.device)

    class_targets = torch.zeros_like(targets_, dtype=torch.long)
    class_targets = torch.where(targets_ <= tol, 0, class_targets)
    class_targets = torch.where(targets_ >= 1 - tol, 2, class_targets)
    in_between = (targets_ > tol) & (targets_ < 1 - tol)
    class_targets = torch.where(in_between, 1, class_targets)

    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(predictions, class_targets)


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
