#  ____                  _           _
# / ___|  ___ __ _ _ __ | |__   ___ | |_
# \___ \ / __/ _` | '_ \| '_ \ / _ \| __|
#  ___) | (_| (_| | | | | |_) | (_) | |_
# |____/ \___\__,_|_| |_|_.__/ \___/ \__|

"""
    habitat_scanbot2d.environments
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Classes to custom a 2d/3d robot scanning environment that follows the RLEnv interface

    :copyright: (c) 2021 by XiXia.
    :license: MIT License, see LICENSE for more details.
"""

from typing import Optional, Dict, Any
from gym import spaces
import numpy as np

import habitat
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.core.embodied_task import Metrics
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_scanbot2d.measures import (
    CompletedArea,
    CompletedRate,
    LongTermGoalReachability,
    ScanningQuality,
    ScanningRate,
    ScanningSuccess,
    PrimitiveActionCount,
)


@baseline_registry.register_env(name="ScanningRLEnv")
class ScanningRLEnv(habitat.RLEnv):
    r"""Reinforcement learning environment which is suitable for training
    autoscanning agent

    Args:
        config: root config node
        dataset: Dataset instance passed to Env
    """

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._normalize_reward_weights()
        self._previous_completed_area: float
        self._previous_scanning_quality: float
        self._previous_primitive_action_count: int
        self._previous_long_term_goal_reachability: float
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)
        self._add_extra_observation_spaces()

    def _add_extra_observation_spaces(self):
        r"""Add extra observation space for SemanticMapBuilder, since
        it's not a Sensor, this design may be changed in future
        """
        try:
            global_semantic_map_space = self.observation_space["semantic_topdown_cuda"]
        except KeyError:
            global_semantic_map_space = self.observation_space["semantic_topdown"]
        self._env.observation_space["global_semantic_map"] = global_semantic_map_space
        if self._core_env_config.TASK.SEMANTIC_MAP_BUILDER.USE_LOCAL_REPRESENTATION:
            scale_factor = (
                self._core_env_config.TASK.SEMANTIC_MAP_BUILDER.MAP_CELL_SIZE
                / self._core_env_config.TASK.SEMANTIC_MAP_BUILDER.LOCAL_MAP_CELL_SIZE
            )
            self._env.observation_space["local_semantic_map"] = spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(
                    int(global_semantic_map_space.shape[0] / scale_factor),
                    int(global_semantic_map_space.shape[1] / scale_factor),
                    global_semantic_map_space.shape[2],
                ),
                dtype=np.float32,
            )

            # When use local representation we need to store the previous
            # global goal position in the observations to help the network
            # figure out the actual goal of its previous action
            self._env.observation_space["previous_goal_position"] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            )

    def reset(self):
        self._episode_success = False
        self._previous_action = None
        observations = super().reset()
        metrics = self._env.get_metrics()
        self._previous_completed_area = metrics[CompletedArea.uuid]
        self._previous_scanning_quality = metrics[ScanningQuality.uuid]
        self._previous_primitive_action_count = metrics[
            PrimitiveActionCount.uuid
        ]
        self._previous_long_term_goal_reachability = metrics[
            LongTermGoalReachability.uuid
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD + self._rl_config.REACHABILITY_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD
            + self._rl_config.EXPLORATION_REWARD
            + self._rl_config.QUALITY_REWARD
            + 1.0,
        )

    def get_reward(self, observations: Observations):
        metrics = self._env.get_metrics()

        current_primitive_action_count = metrics[PrimitiveActionCount.uuid]
        reward = (
            current_primitive_action_count - self._previous_primitive_action_count
        ) * self._slack_reward
        self._previous_primitive_action_count = current_primitive_action_count

        current_completed_area = metrics[CompletedArea.uuid]
        reward += (
            current_completed_area - self._previous_completed_area
        ) * self._exploration_reward
        self._previous_completed_area = current_completed_area

        current_scanning_quality = metrics[ScanningQuality.uuid]
        reward += (
            current_scanning_quality - self._previous_scanning_quality
        ) * self._quality_reward
        self._previous_scanning_quality = current_scanning_quality

        current_long_term_goal_reachability = metrics[LongTermGoalReachability.uuid]
        reward += (
            current_long_term_goal_reachability
            - self._previous_long_term_goal_reachability
        ) * self._reachability_reward
        self._previous_long_term_goal_reachability = current_long_term_goal_reachability

        if metrics[ScanningSuccess.uuid]:
            reward += self._success_reward
            self._episode_success = True

        return reward

    def _updates_achieved_rate(self, metrics: Metrics):
        completed_rate = metrics[CompletedRate.uuid]
        try:
            scanning_rate = metrics[ScanningRate.uuid]
        except KeyError:
            scanning_rate = None
        self._env._task.measurements.measures[
            ScanningSuccess.uuid
        ].update_statistics(  # type: ignore
            completed_rate,
            scanning_rate,
        )

    def get_done(self, observations):
        if self._env.episode_over or self._episode_success:
            self._updates_achieved_rate(self._env.get_metrics())
            return True
        return False

    def get_info(self, observations):
        result = self._env.get_metrics()
        result["scene_id"] = self._env.current_episode.scene_id
        return result

    def get_achieved_rate_statistics(self) -> Dict[str, Any]:
        return self._env._task.measurements.measures[
            ScanningSuccess.uuid
        ].achieved_rate_statistics  # type: ignore

    def set_reachability_reward(self, percentage) -> float:
        self._reachability_reward = self._rl_config.REACHABILITY_REWARD * percentage
        return self._reachability_reward

    def _normalize_reward_weights(self):
        map_size_in_meters = (
            self._core_env_config.TASK.SEMANTIC_MAP_BUILDER.MAP_SIZE_IN_METERS
        )
        self._slack_reward = self._rl_config.SLACK_REWARD
        self._reachability_reward = (
            self._rl_config.REACHABILITY_REWARD / map_size_in_meters
        )
        self._exploration_reward = self._rl_config.EXPLORATION_REWARD
        self._success_reward = self._rl_config.SUCCESS_REWARD
        self._quality_reward = self._rl_config.QUALITY_REWARD
