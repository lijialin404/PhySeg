# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file contains only the utility layers used by the image-only SAM 2
# backbone and decoder.

from __future__ import annotations

import torch
from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return inputs
        keep_prob = 1.0 - self.drop_prob
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_tensor = inputs.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return inputs * random_tensor


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: type[nn.Module] = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(
                [input_dim, *hidden],
                [*hidden, output_dim],
            )
        )
        self.activation = activation()
        self.sigmoid_output = sigmoid_output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for index, layer in enumerate(self.layers):
            output = (
                self.activation(layer(output))
                if index < len(self.layers) - 1
                else layer(output)
            )
        return torch.sigmoid(output) if self.sigmoid_output else output


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(1, keepdim=True)
        variance = (inputs - mean).pow(2).mean(1, keepdim=True)
        normalized = (inputs - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]
