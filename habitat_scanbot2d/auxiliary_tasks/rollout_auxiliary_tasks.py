#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Optional
import math

import torch
import torch.nn as nn

from habitat_baselines.common.baseline_registry import baseline_registry


def register_auxiliary_task(to_register=None, *, name: Optional[str] = None):
    r"""Register an auxiliary task
    :param name: Key with which the env will be registered.
        If None will use the name of the class.
    """
    return baseline_registry._register_impl(
        "auxiliary_task", to_register, name, assert_type=RolloutAuxiliaryTask
    )


class RolloutAuxiliaryTask(nn.Module):
    r"""Rollout-based self-supervised auxiliary task base class."""

    def __init__(self, ppo_config, auxiliary_config, device, **kwargs):
        super().__init__()
        self.ppo_config = ppo_config
        self.auxiliary_config = auxiliary_config
        self.device = device

    def forward(self, *x):
        raise NotImplementedError

    @abc.abstractmethod
    def get_loss(self, map_feature, **kwargs) -> torch.Tensor:
        pass


@register_auxiliary_task(name="PathComplexity")
class PathComplexityAuxiliaryTask(RolloutAuxiliaryTask):
    def __init__(
        self,
        ppo_config,
        auxiliary_config,
        device,
        **kwargs,
    ):
        super().__init__(ppo_config, auxiliary_config, device, **kwargs)
        self.step_count_class_num = math.ceil(
            (auxiliary_config.step_count_clip + 1)
            / auxiliary_config.step_count_class_interval
        )
        self.fc = nn.Linear(
            ppo_config.hidden_size + 32,
            2 * self.step_count_class_num,
        ).to(device)
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def get_loss(
        self,
        map_feature: torch.Tensor,
        action_embedding: torch.Tensor = None,
        left_step_ground_truth: torch.Tensor = None,
        right_step_ground_truth: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        assert (
            action_embedding is not None
            and left_step_ground_truth is not None
            and right_step_ground_truth is not None
        ), f"{type(self)} requires actions, left_step_ground_truth and right_step_ground_truth!"
        feature = torch.cat((map_feature, action_embedding), dim=1)
        out = self.fc(feature)
        loss = self.loss_fn(
            out[:, : self.step_count_class_num],
            torch.squeeze(left_step_ground_truth, dim=1),
        ) + self.loss_fn(
            out[:, self.step_count_class_num :],
            torch.squeeze(right_step_ground_truth, dim=1),
        )
        return loss / 2


@register_auxiliary_task(name="QualityMemory")
class QualityMemoryAuxTask(RolloutAuxiliaryTask):
    def __init__(
        self,
        ppo_config,
        auxiliary_config,
        device,
        num_recurrent_layers=1,
        **kwargs,
    ):
        super().__init__(ppo_config, auxiliary_config, device, **kwargs)
        self._num_steps = ppo_config.num_steps
        self.fc = nn.Sequential(
            nn.Linear(
                ppo_config.hidden_size + ppo_config.hidden_size * num_recurrent_layers,
                1,
            ).to(device),
            nn.Tanh(),
        )
        self.loss_fn = nn.MSELoss()

    def get_loss(
        self,
        map_feature,
        recurrent_hidden_states: torch.Tensor = None,
        quality_increase_ratio_ground_truth: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        assert (
            recurrent_hidden_states is not None
            and quality_increase_ratio_ground_truth is not None
        ), f"QualityMemoryAuxTask requires recurrent_hidden_states and scanning_quality!"
        num_environments = quality_increase_ratio_ground_truth.shape[0] // self._num_steps
        quality_increase_ratio_ground_truth = quality_increase_ratio_ground_truth[
            0 : (self._num_steps - 1) * num_environments, ...
        ]
        recurrent_hidden_states = recurrent_hidden_states.flatten(0, 1).flatten(-2, -1)
        map_feature = map_feature[0 : (self._num_steps - 1) * num_environments, ...]
        feature = torch.cat(
            (map_feature, recurrent_hidden_states),
            dim=1,
        )
        out = self.fc(feature)
        out = (out + 1) / 2
        loss = self.loss_fn(out, quality_increase_ratio_ground_truth)
        return loss
