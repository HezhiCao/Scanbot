#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Tuple, List

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F
import copy

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.utils.common import CustomNormal, GaussianNet
from habitat_baselines.rl.ppo import Net, Policy
from habitat_scanbot2d.policies import encoders
from habitat_scanbot2d.sensors import MapPoseSensor, SemanticTopDownSensor
from habitat_scanbot2d.auxiliary_tasks.rollout_auxiliary_tasks import (
    RolloutAuxiliaryTask,
)
from habitat_scanbot2d.navigation_action import NavigationAction
from habitat_scanbot2d.cuda_sensors import SemanticTopDownCudaSensor


@baseline_registry.register_policy
class ScanningGlobalPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        base_channels: int = 64,
        visual_backbone: str = "resnet18",
        map_backbone: str = "simple_net",
        normalize_visual_inputs: bool = False,
        policy_config: Config = None,
        used_inputs: List[str] = [],
        auxiliary_tasks: List[RolloutAuxiliaryTask] = [],
        **kwargs
    ):
        if policy_config is not None:
            discrete_actions = policy_config.action_distribution_type == "categorical"
        else:
            discrete_actions = True

        super().__init__(
            ScanningGlobalNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                visual_backbone=visual_backbone,
                map_backbone=map_backbone,
                base_channels=base_channels,
                normalize_visual_inputs=normalize_visual_inputs,
                discrete_actions=discrete_actions,
                used_inputs=used_inputs,
            ),
            dim_actions=(
                action_space.n if discrete_actions else action_space.shape[0]
            ),  # for action distribution
            policy_config=policy_config,
        )
        self.auxiliary_tasks = auxiliary_tasks
        # Try to use observations independent std
        if (
            self.action_distribution_type == "gaussian"
            and policy_config.ACTION_DIST.use_independent_std
        ):
            self.action_distribution = GaussianNetWithIndependentStd(
                self.net.output_size,
                self.dim_actions,
                policy_config.ACTION_DIST,
            )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space, **kwargs
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            visual_backbone=config.RL.DDPPO.backbone,
            global_backbone=config.RL.PPO.global_backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            policy_config=config.RL.POLICY,
            used_inputs=config.RL.PPO.used_inputs,
            auxiliary_tasks=kwargs["auxiliary_tasks"],
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if self.action_distribution_type == "gaussian":
            self.mean = distribution.mean
            self.stddev = distribution.stddev

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)  # type: ignore

        return value, action, action_log_probs, rnn_hidden_states

    def _construct_auxiliary_task_input(self, batch, num_steps):
        assert (
            self.net.use_semantic_map
        ), "ScanningGlobalPolicy._construct_auxiliary_task_input requires semantic_map_encoder!"
        observations = batch["observations"]
        semantic_map_feature = self.net.semantic_map_encoder(observations)
        try:
            scale_factor = int(
                observations[SemanticTopDownCudaSensor.uuid].shape[1]
                / observations["global_semantic_map"].shape[1]
            )
        except KeyError:
            scale_factor = int(
                observations[SemanticTopDownSensor.uuid].shape[1]
                / observations["global_semantic_map"].shape[1]
            )
        actions_embedding = self.net.prev_action_embedding(
            torch.clip(
                torch.clip(batch["actions"], min=-1.0, max=1.0) / scale_factor
                + batch["observations"]["map_pose"][:, :2],
                min=-1.0,
                max=1.0,
            )
        )

        num_environments = batch["masks"].shape[0] // num_steps
        recurrent_hidden_states = batch["recurrent_hidden_states"]
        for i in range(num_steps):
            if i <= 1:
                _, recurrent_hidden_states = self.net(
                    observations[i * num_environments : (i + 1) * num_environments, ...],
                    recurrent_hidden_states,
                    batch["prev_actions"][
                        i * num_environments : (i + 1) * num_environments, ...
                    ],
                    batch["masks"][i * num_environments : (i + 1) * num_environments, ...],
                )
                recurrent_hidden_states_l = recurrent_hidden_states.unsqueeze(dim=0)
            else:
                _, recurrent_hidden_states = self.net(
                    observations[i * num_environments : (i + 1) * num_environments, ...],
                    recurrent_hidden_states,
                    batch["prev_actions"][
                        i * num_environments : (i + 1) * num_environments, ...
                    ],
                    batch["masks"][i * num_environments : (i + 1) * num_environments, ...],
                )
                recurrent_hidden_states_l = torch.cat(
                    (
                        recurrent_hidden_states_l,
                        recurrent_hidden_states.unsqueeze(dim=0),
                    ),
                    dim=0,
                )
        return semantic_map_feature, actions_embedding, recurrent_hidden_states_l

    def compute_auxiliary_task_losses(
        self, batch, num_steps, **kwargs
    ) -> List[torch.Tensor]:
        if len(self.auxiliary_tasks) == 0:
            return []
        (
            semantic_map_feature,
            actions_embedding,
            recurrent_hidden_states_l,
        ) = self._construct_auxiliary_task_input(batch, num_steps)
        return [
            task.get_loss(
                semantic_map_feature,
                action_embedding=actions_embedding,
                left_step_ground_truth=batch["left_step_count"],
                right_step_ground_truth=batch["right_step_count"],
                recurrent_hidden_states=recurrent_hidden_states_l,
                quality_increase_ratio_ground_truth=batch["quality_increase_ratio"],
            )
            for task in self.auxiliary_tasks
        ]


class VisualEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        base_channels: int = 32,
        num_groups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
        used_inputs: List[str] = [],
    ):
        super().__init__()

        if "rgb" in used_inputs and "rgb" in observation_space.spaces:
            self._num_rgb_channels = observation_space.spaces["rgb"].shape[2]
            spatial_size = observation_space.spaces["rgb"].shape[0] // 2
        else:
            self._num_rgb_channels = 0

        if "depth" in used_inputs and "depth" in observation_space.spaces:
            self._num_depth_channels = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0] // 2
        else:
            self._num_depth_channels = 0

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._num_rgb_channels + self._num_depth_channels
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        input_channels = self._num_rgb_channels + self._num_depth_channels
        assert input_channels != 0

        self.backbone = make_backbone(input_channels, base_channels, num_groups)

        final_spatial = int(spatial_size * self.backbone.final_spatial_compress)

        after_compression_flat_size = 2048
        num_compression_channels = int(
            round(after_compression_flat_size / (final_spatial ** 2))
        )
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
        )

        self.output_shape = (
            num_compression_channels,
            final_spatial,
            final_spatial,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_input = []
        if self._num_rgb_channels > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations.float() / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._num_depth_channels > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class GlobalEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        base_channels: int = 32,
        num_groups: int = 32,
        output_size: int = 512,
        make_backbone=None,
    ):
        super().__init__()

        assert (
            "global_semantic_map" in observation_space.spaces
        ), "global_semantic_map observation is needed for GlobalEncoder"

        input_channels = observation_space.spaces["global_semantic_map"].shape[2]
        spatial_shape = observation_space.spaces["global_semantic_map"].shape[:2]

        self.backbone = make_backbone(
            spatial_shape,
            input_channels,
            base_channels,
            num_groups,
            output_size,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        semantic_observations = observations["global_semantic_map"]
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        x = semantic_observations.permute(0, 3, 1, 2)
        x = self.backbone(x)
        return x


class ScanningGlobalNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        visual_backbone,
        map_backbone,
        base_channels,
        normalize_visual_inputs: bool,
        discrete_actions: bool = True,
        used_inputs: list = [],
    ):
        super().__init__()

        self.discrete_actions = discrete_actions
        self.used_inputs = used_inputs
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        else:
            self.prev_action_embedding = nn.Linear(action_space.shape[0], 32)

        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action

        if MapPoseSensor.uuid in observation_space.spaces:
            input_map_pose_dim = observation_space.spaces[MapPoseSensor.uuid].shape[0]
            self.map_pose_embedding = nn.Linear(input_map_pose_dim, 32)
            rnn_input_size += 32

        if "rgb" in self.used_inputs or "depth" in self.used_inputs:
            self.visual_encoder = VisualEncoder(
                observation_space,
                base_channels=base_channels,
                num_groups=base_channels // 2,
                make_backbone=getattr(resnet, visual_backbone),
                normalize_visual_inputs=normalize_visual_inputs,
                used_inputs=used_inputs,
            )

            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
                    nn.ReLU(True),
                )
        else:
            self.visual_encoder = None

        if "semantic_map" in self.used_inputs:
            self.semantic_map_encoder = GlobalEncoder(
                observation_space,
                base_channels=base_channels,
                num_groups=base_channels // 2,
                output_size=hidden_size,
                make_backbone=getattr(encoders, map_backbone),
            )
        else:
            self.semantic_map_encoder = None

        if not self.is_blind and self.use_semantic_map:
            self._hidden_size = 2 * hidden_size
        else:
            self._hidden_size = hidden_size

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind and not self.use_semantic_map else self._hidden_size)
            + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder is None or self.visual_encoder.is_blind

    @property
    def use_semantic_map(self):
        return self.semantic_map_encoder is not None

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        if self.use_semantic_map:
            semantic_feats = self.semantic_map_encoder(observations)
            x.append(semantic_feats)

        if MapPoseSensor.uuid in observations:
            map_pose = observations[MapPoseSensor.uuid]
            x.append(self.map_pose_embedding(map_pose))

        # Use need to compute prev_actions in global coordinate
        # instead of local coordinate, this value will be filled
        # in NavigationAction
        if "previous_goal_position" in observations:
            prev_actions = self.prev_action_embedding(
                masks * observations["previous_goal_position"].float()
            )
        else:
            if self.discrete_actions:
                prev_actions = prev_actions.squeeze(-1)
                start_token = torch.zeros_like(prev_actions)
                prev_actions = self.prev_action_embedding(
                    torch.where(masks.view(-1), prev_actions + 1, start_token)
                )
            else:
                prev_actions = self.prev_action_embedding(masks * prev_actions.float())

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(out, rnn_hidden_states, masks)

        return out, rnn_hidden_states


class GaussianNetWithIndependentStd(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        config: Config,
    ) -> None:
        super().__init__()

        self.action_activation = config.action_activation
        self.use_log_std = config.use_log_std
        self.use_softplus = config.use_softplus
        if config.use_log_std:
            self.min_std = config.min_log_std
            self.max_std = config.max_log_std
        else:
            self.min_std = config.min_std
            self.max_std = config.max_std

        self.mu = nn.Linear(num_inputs, num_outputs)
        self.std = nn.parameter.Parameter(torch.zeros(num_outputs))

        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0)

    def forward(self, x: torch.Tensor) -> CustomNormal:
        mu = self.mu(x)
        if self.action_activation == "tanh":
            mu = torch.tanh(mu)

        std = torch.clamp(self.std, min=self.min_std, max=self.max_std)
        if self.use_log_std:
            std = torch.exp(std)
        if self.use_softplus:
            std = torch.nn.functional.softplus(std)

        return CustomNormal(mu, std)
