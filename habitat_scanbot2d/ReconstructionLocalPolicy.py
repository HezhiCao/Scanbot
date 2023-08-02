import math

import torch
from torch import nn
import numpy as np
from gym import spaces
from typing import Optional, Dict

from habitat.config import Config
from habitat_baselines.rl.ppo import Policy, Net
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder

from habitat_scanbot2d.policies import encoders


@baseline_registry.register_policy
class ReconstructionLocalPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        cnn_baseplanes: int = 32,
        backbone: str = "voxel_net",
        policy_config: Config = None,
        **kwargs
    ):
        if policy_config is not None:
            discrete_actions = policy_config.action_distribution_type == "categorical"
            self.action_distribution_type = policy_config.action_distribution_type
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"
        super(ReconstructionLocalPolicy, self).__init__(
            ReconstructionLocalNet(
                observation_space,
                action_space,
                hidden_size,
                num_recurrent_layers,
                rnn_type,
                voxel_backbone=backbone,
                cnn_baseplanes=cnn_baseplanes,
                discrete_actions=discrete_actions,
            ),
            dim_actions=action_space.n,
            policy_config=policy_config,
        )

    @classmethod
    def from_config(cls, config, observation_space, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            rnn_type=config.RL.DDPPO.rnn_type,
            backbone=config.RL.PPO.voxel_backbone,
            policy_config=config.RL.POLICY,
        )


class ReconstructionLocalNet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layer: int,
        rnn_type: str,
        voxel_backbone,
        cnn_baseplanes,
        discrete_actions: bool = False,
    ):
        super(ReconstructionLocalNet, self).__init__()
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        else:
            self.prev_action_embedding = nn.Linear(action_space.n, 32)
        rnn_input_size = 32
        self.voxel_encoder = CNNEncoder(
            observation_space,
            baseplanes=cnn_baseplanes,
            ngroups=cnn_baseplanes // 2,
            make_backbone=getattr(encoders, voxel_backbone),
        )
        self._hidden_size = hidden_size
        if not self.voxel_encoder.is_blind:
            self.voxel_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.voxel_encoder.output_shape)),
                hidden_size,
                nn.ReLU(inplace=True),
            )
        self.state_encoder = build_rnn_state_encoder(
            (0 if self.voxel_encoder.is_blind else hidden_size) + rnn_input_size,
            hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layer,
        )
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ):
        x = []
        feature = self.voxel_encoder(observations["objects_voxel"])
        feature = self.voxel_fc(feature)
        x.append(feature)
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


class CNNEncoder(nn.Module):
    def __init__(
        self,
        observavtion_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        make_backbone=None,
    ):
        super().__init__()

        if not self.is_blind:
            self.input_channels = observavtion_space.spaces["objects_voxel"].shape[0]
            spatial_size = observavtion_space.spaces["objects_voxel"].shape[1]
            self.backbone = make_backbone(self.input_channels, baseplanes, ngroups)
            final_spatial = math.ceil(
                spatial_size * self.backbone.final_spatial_compress
            )
            after_compression_flat_size = 1024
            num_compression_channels = round(
                after_compression_flat_size / (final_spatial ** 3)
            )
            assert num_compression_channels >= 1, print(
                "Parameter num_compression_channels in CNNEncoder set too small !"
            )
            self.compression = nn.Sequential(
                nn.Conv3d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(inplace=True),
            )
            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
                final_spatial,
            )

    @property
    def is_blind(self):
        return self.input_channels == 0

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.is_blind:
            return None
        # [BATCH X CHANNEL X DEPTH X HEIGHT X WIDTH]
        x = self.backbone(observations["objects_voxel"])
        x = self.compression(x)
        return x
