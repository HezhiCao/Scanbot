#  ____                  _           _
# / ___|  ___ __ _ _ __ | |__   ___ | |_
# \___ \ / __/ _` | '_ \| '_ \ / _ \| __|
#  ___) | (_| (_| | | | | |_) | (_) | |_
# |____/ \___\__,_|_| |_|_.__/ \___/ \__|

"""
    habitat_scanbot2d.scanning_task
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Definition of scanning task related Episode and Task

    :copyright: (c) 2021 by XiXia.
    :license: MIT License, see LICENSE for more details.
"""

from typing import Any, Optional, Dict, Union, cast

import attr
import torch
import numpy as np

from habitat.core.embodied_task import EmbodiedTask
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.dataset import Dataset, Episode
from habitat.core.simulator import Observations, Simulator
from habitat.tasks.nav.nav import merge_sim_episode_config
from habitat_scanbot2d.semantic_map_builder import SemanticMapBuilder
from habitat_scanbot2d.navigation_quality_map_builder import (
    NavigationQualityMapBuilder,
)


@attr.s(auto_attribs=True, kw_only=True)
class ScanningEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, maximum collision count and minimum completion
    rate. An episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        start_room: room id
        max_collision_times: maximum allowed collision times
        min_completion_rate: minimum completion rate that a episode is considered
            to be successful
    """

    start_room: Optional[str] = None


@registry.register_task(name="Scanning-v0")
class ScanningTask(EmbodiedTask):
    r"""Class for scanning oriented task, this task contains several more attributes
    than its base class:
        * observations: observations from previous step
        * current_episode: Episode instance for current step

    Args:
        config: config for the task
        sim: reference to the simulator passed to the base class
        dataset: reference to dataset passed to the base class
    """

    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ):
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.semantic_map_builder = SemanticMapBuilder(config.SEMANTIC_MAP_BUILDER)
        if "QUALITY_INCREASE_RATIO" in self._config.MEASUREMENTS:
            self.navigation_quality_map_builder = NavigationQualityMapBuilder(
                config.NAVIGATION_QUALITY_MAP_BUILDER
            )
        self.observations: Observations
        self.current_episode: Episode
        self.primitive_actions_in_last_navigation = 0
        self.left_step_count = 0
        self.right_step_count = 0
        self.navigation_quality_count = 0
        self.navigation_quality_increase = 0

    def step(self, action: Dict[str, Any], episode: Episode) -> Observations:
        self.current_episode = episode
        self._before_step(action)
        self.observations = super().step(action, episode)
        self._after_step(action)
        return self.observations

    def reset(self, episode: Episode):
        self.current_episode = episode
        self.observations = super().reset(episode)
        self.semantic_map_builder.reset()
        self.semantic_map_builder.update(self.observations)
        # Initially scan the surroundings
        self.actions["NAVIGATION"].look_around(self)
        # Initialize the observations with an (0, 0) previous goal position
        # since NavigationAction will only insert this in step
        self.observations["previous_goal_position"] = np.zeros((2,), dtype=np.float32)
        return self.observations

    def _before_step(self, action):
        if action["action"] == "NAVIGATION":
            self.primitive_actions_in_last_navigation = 0
            if "QUALITY_INCREASE_RATIO" in self._config.MEASUREMENTS:
                self.navigation_quality_map_builder.reset(self.observations)
        elif action["action"] == "MOVE_FORWARD":
            self.primitive_actions_in_last_navigation += 1

    def _after_step(self, action):
        if "QUALITY_INCREASE_RATIO" in self._config.MEASUREMENTS:
            if action["action"] != "NAVIGATION":
                self.navigation_quality_map_builder.update(self.observations)
            else:
                self.quality_increase_ratio = (
                    self.navigation_quality_map_builder.compute_quality_increase_ratio(self.observations)
                )
        if "OBJECT_DISCOVERY" in self._config.MEASUREMENTS:
            self.measurements.measures["object_discovery"].update_metric(observations=self.observations)

        self.semantic_map_builder.update(self.observations)

    def overwrite_sim_config(self, sim_config: Config, episode: Episode) -> Config:
        return merge_sim_episode_config(sim_config, episode)

    def transfer_observations_to_cpu(self) -> Observations:
        result = {}
        for sensor_name, observation in self.observations.items():
            if isinstance(observation, torch.Tensor):
                result[sensor_name] = observation.cpu().numpy()
            else:
                result[sensor_name] = observation
        return cast(Observations, result)

    def _check_episode_is_active(
        self,
        *args: Any,
        action: Union[int, Dict[str, Any]],
        episode: Episode,
        **kwargs: Any,
    ) -> bool:
        return True
