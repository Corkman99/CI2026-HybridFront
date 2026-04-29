import numpy as np
import torch
import xarray as xr

from notebooks.sunquist_classifier import (
    ClassificationNetwork,
    SundquistClassifier,
    estimate_cross_entropy,
)


def test_classification_network_passes_selected_inputs():
    batch_size = 2
    num_channels = 4
    levels = 7
    height = 8
    width = 8

    input_level = torch.randn(batch_size, num_channels, levels, height, width)
    input_auxiliary = torch.randn(batch_size, 3, height, width)

    network = ClassificationNetwork()
    logits = network(input_level, input_auxiliary)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, 3, height, width)
    assert torch.isfinite(logits).all()


def test_estimate_cross_entropy_accepts_xarray_targets():
    batch_size = 2
    num_classes = 3
    height = 8
    width = 8

    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.rand(batch_size, height, width)

    loss = estimate_cross_entropy(predictions, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_sundquist_classifier_forward_returns_single_channel_tensor():
    batch_size = 1
    input_level = torch.randn(batch_size, 3, 7, 8, 8)
    input_auxiliary = torch.randn(batch_size, 2, 8, 8)

    model = SundquistClassifier()
    output = model(input_level, input_auxiliary)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 1, 8, 8)
    assert torch.all((output >= 0.0) & (output <= 1.0))


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
