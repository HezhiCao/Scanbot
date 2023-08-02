import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.common.rollout_storage import RolloutStorage


class AuxiliaryRoulloutStorage(RolloutStorage):
    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        action_shape: Optional[Tuple[int]] = None,
        is_double_buffered: bool = False,
        discrete_actions: bool = True,
    ):
        super(AuxiliaryRoulloutStorage, self).__init__(
            numsteps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
            num_recurrent_layers,
            action_shape,
            is_double_buffered,
            discrete_actions,
        )
        self.buffers["left_step_count"] = torch.zeros(numsteps + 1, num_envs, 1).long()
        self.buffers["right_step_count"] = torch.zeros(numsteps + 1, num_envs, 1).long()
        self.buffers["quality_increase_ratio"] = torch.zeros(
            numsteps + 1, num_envs, 1
        ).float()

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        left_step_count: Optional[torch.Tensor] = None,
        right_step_count: Optional[torch.Tensor] = None,
        quality_increase_ratio: Optional[torch.Tensor] = None,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
            left_step_count=left_step_count,
            right_step_count=right_step_count,
            quality_increase_ratio=quality_increase_ratio,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )
