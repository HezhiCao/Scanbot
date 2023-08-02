#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
from torch import Tensor
from torch import nn
import numpy as np
import torch


class SimpleNet(nn.Module):
    def __init__(
        self,
        input_spatial_shape: Tuple[int, int],
        input_channels: int,
        base_channels: int,
        num_groups: int,
        output_size,
    ) -> None:
        super(SimpleNet, self).__init__()
        kernel_sizes = np.array([8, 4, 4, 3, 3], dtype=np.int32)
        strides = np.array([4, 2, 2, 1, 1], dtype=np.int32)
        output_spatial_shape = self.compute_output_spatial_shape(
            input_spatial_shape,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=np.zeros((5,), dtype=np.int32),
            dilations=np.ones((5,), dtype=np.int32),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(
                input_channels,
                base_channels,
                kernel_sizes[0],
                stride=strides[0],
                bias=False,
            ),
            nn.GroupNorm(num_groups, base_channels),
            nn.ReLU(True),
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_sizes[1],
                stride=strides[1],
                bias=False,
            ),
            nn.GroupNorm(num_groups, base_channels * 2),
            nn.ReLU(True),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 2,
                kernel_sizes[2],
                stride=strides[2],
                bias=False,
            ),
            nn.GroupNorm(num_groups, base_channels * 2),
            nn.ReLU(True),
            nn.Conv2d(
                base_channels * 2,
                base_channels,
                kernel_sizes[3],
                stride=strides[3],
                bias=False,
            ),
            nn.GroupNorm(num_groups, base_channels),
            nn.ReLU(True),
            nn.Conv2d(
                base_channels,
                base_channels // 2,
                kernel_sizes[4],
                stride=strides[4],
                bias=False,
            ),
            nn.GroupNorm(num_groups // 2, base_channels // 2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(
                base_channels // 2 * output_spatial_shape[0] * output_spatial_shape[1],
                output_size * 2,
            ),
            nn.ReLU(True),
            nn.Linear(
                output_size * 2,
                output_size,
            ),
            nn.ReLU(True),
        )

        self.layer_init()

    @staticmethod
    def compute_output_spatial_shape(
        input_shape: Tuple[int, int],
        kernel_sizes: np.ndarray,
        strides: np.ndarray,
        paddings: np.ndarray,
        dilations: np.ndarray,
    ) -> Tuple[int, int]:
        r"""Calculates the output height and width based on the input
        height and width of the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        # assert len(input_shape) == 2
        output_shape = list(input_shape)
        for i in range(len(kernel_sizes)):
            for j in range(len(input_shape)):
                output_shape[j] = int(
                    np.floor(
                        (
                            (
                                output_shape[j]
                                + 2 * paddings[i]
                                - dilations[i] * (kernel_sizes[i] - 1)
                                - 1
                            )
                            / strides[i]
                        )
                        + 1
                    )
                )

        return tuple(output_shape)  # type: ignore

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)  # type: ignore

    def forward(self, x) -> Tensor:
        x = self.cnn(x)
        return x


def simple_net(
    spatial_shape: Tuple[int, int],
    input_channels: int,
    base_channels: int,
    num_groups: int,
    output_size: int,
):
    model = SimpleNet(
        spatial_shape, input_channels, base_channels, num_groups, output_size
    )
    return model


class VoxelNet(nn.Module):
    def __init__(
        self,
        input_spatial_shape: np.ndarray,
        in_channels: int,
        base_planes: int,
        ngroups: int,
        output_size,
    ):
        super(VoxelNet, self).__init__()
        kernel_sizes = np.array([5, 5, 5, 3, 3], dtype=np.int32)
        strides = np.array([2, 2, 2, 1, 1], dtype=np.int32)
        paddings = np.floor(kernel_sizes/2).astype(np.int32)
        channels = [
            in_channels,
            base_planes,
            base_planes * 2,
            base_planes * 2 * 2,
            base_planes * 2,
            base_planes,
        ]
        output_spatial_shape = SimpleNet.compute_output_spatial_shape (
            input_spatial_shape,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            dilations=np.ones((5,), dtype=np.int32),
        )
        modules = []
        for i in range(len(kernel_sizes)):
            modules.append(
                nn.Conv3d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                )
            )
            modules.append(nn.GroupNorm(ngroups, channels[i + 1]))
            modules.append(nn.ReLU(inplace=True))
        self.cnn = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Linear(np.prod(output_spatial_shape) * channels[-1], output_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(
                output_size * 2,
                output_size,
            ),
            nn.ReLU(True),
        )
        self.layer_init()

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, x) -> Tensor:
        x = x.float()
        # for i, layer in enumerate(self.cnn):
        #     x = layer(x)
        #     if isinstance(layer, (nn.Conv3d, nn.Linear)):
        #         layer.weight.retain_grad()
        #         if layer.weight.grad is not None:
        #             print("layer %d:\t max_grad: %.18f" % (i, torch.max(layer.weight.grad)))
        # return x
        return self.cnn(x)



def voxel_net(input_spatial_shape, in_channels, base_planes, ngroups, output_size):
    model = VoxelNet(
        input_spatial_shape, in_channels, base_planes, ngroups, output_size
    )
    return model
