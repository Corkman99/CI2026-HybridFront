#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging

# External modules
import torch

# Internal modules


main_logger = logging.getLogger(__name__)


class InputNormalisation(torch.nn.Module):
    r"""
    Normalises input tensors by a predefined mean and standard
    deviation, stored as non-trainable buffers.

    Mean and standard deviation are registered as buffers so they
    are included in ``state_dict()``, transferred with ``.to()``,
    and serialised with the network checkpoint — but never updated
    by the optimiser.

    Parameters
    ----------
    mean : torch.Tensor
        Per-channel mean. Must be broadcastable with the input.
    std : torch.Tensor
        Per-channel standard deviation. Must be broadcastable
        with the input.
    eps : float, optional, default = 1e-6
        Small constant added to ``std`` to avoid division by zero.

    Attributes
    ----------
    mean : torch.Tensor
        Registered buffer holding the normalisation mean.
    std : torch.Tensor
        Registered buffer holding the normalisation std.
    eps : float
        Numerical stability constant.

    Examples
    --------
    >>> mean = torch.zeros(3, 1, 1)
    >>> std = torch.ones(3, 1, 1)
    >>> norm = InputNormalisation(mean, std)
    >>> x = torch.randn(8, 3, 64, 64)
    >>> norm(x).shape
    torch.Size([8, 3, 64, 64])
    """

    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Normalise ``x`` to zero mean and unit variance.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, broadcastable with ``self.mean`` and
            ``self.std``.

        Returns
        -------
        torch.Tensor
            Normalised tensor of the same shape as ``x``.
        """
        return (x - self.mean) / (self.std + self.eps)


# Copilot-written
class PerPixelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, H, W, bias=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features, H, W))
        self.bias = (
            torch.nn.Parameter(torch.zeros(out_features, H, W)) if bias else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features, H, W)
        out = torch.einsum("bihw,iohw->bohw", x, self.weight)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        return out


class PixelWiseClassifier(torch.nn.Module):
    def __init__(
        self, in_channels=3, levels=7, aux_channels=0, hidden=64, num_classes=3
    ):
        super().__init__()
        self.feat_dim = in_channels * levels + aux_channels
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(self.feat_dim, hidden, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, hidden, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L*in_channels + aux_channels, H, W)
        x = x.view(x.size(0), self.feat_dim, x.size(-2), x.size(-1))
        logits = self.net(x)  # (B, num_classes, H, W)
        return logits
