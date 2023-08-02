#  ____                  _           _
# / ___|  ___ __ _ _ __ | |__   ___ | |_
# \___ \ / __/ _` | '_ \| '_ \ / _ \| __|
#  ___) | (_| (_| | | | | |_) | (_) | |_
# |____/ \___\__,_|_| |_|_.__/ \___/ \__|

"""
    habitat_scanbot2d.navigation_action
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    A class that implement a high-level navigation action, which
    decomposes a goal point into a series of primitive action

    :copyright: (c) 2021 by XiXia.
    :license: MIT License, see LICENSE for more details.
"""


from collections import defaultdict
from typing import Any, Optional, List, Dict, Union
from gym import spaces
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.dataset import Episode
from habitat.core.embodied_task import EmbodiedTask, SimulatorTaskAction
from habitat.core.simulator import Observations, Simulator
from habitat.utils import profiling_wrapper
from habitat_scanbot2d.primitive_action_planner import PrimitiveActionPlanner
from habitat_scanbot2d.astar_path_finder import AStarPathFinder
from habitat_scanbot2d.sensors import (
    SemanticTopDownSensor,
    MapPoseSensor,
)
from habitat_scanbot2d.cuda_sensors import SemanticTopDownCudaSensor
from habitat_scanbot2d.utils.visualization import SemanticMapViewer

from habitat_scanbot2d.scanning_task import ScanningTask


@registry.register_task_action
class NavigationAction(SimulatorTaskAction):
    r"""This action performs a sequence of primitive actions from a given start map position
    to goal map position

    Args:
        config: config for the action
        sim: reference to the simulator passed to base class
        visualization: whether to visualize the agent movement
    """

    name: str = "NAVIGATION"

    def __init__(
        self,
        *args: Any,
        config: Config,
        sim: Simulator,
        visualization=False,
        **kwargs: Any,
    ):
        self.config = config
        self.path_planner = AStarPathFinder()
        self.primitive_step_count = 0
        self.collided_count = 0
        self.trapped_count = 0
        self.last_planning_step = 0
        self.use_local_representation = False
        self.goal_position: np.ndarray
        self.current_map_pose: np.ndarray
        self.original_obstacle_map: torch.Tensor
        self.normalized_obstacle_map: torch.Tensor
        self.previous_explored_map: np.ndarray
        self.current_explored_map: np.ndarray
        self._current_episode_id = None
        if self.config.USE_SIMULATOR_ORACLE:
            self.collided_position_counter = defaultdict(lambda: 0)
            self.collided_positions = []
        else:
            self.collided_poses = []
        self.visualization = visualization or config.VISUALIZATION
        if self.visualization:
            plt.ion()
            self._semantic_map_viewer: SemanticMapViewer
        super().__init__(self, *args, config=config, sim=sim, **kwargs)
        self._initialize_obstacle_inflation()

    @property
    def action_space(self):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    @profiling_wrapper.RangeContext("NavigationAction.step")
    def step(
        self,
        *args,
        task: ScanningTask,
        goal_position: np.ndarray,
        **kwargs,
    ) -> Observations:
        r"""Step method called from EmbodiedTask, this action assumes the task contains
        observations and current_episode attribute, which holds latest observations.
        It will also call task.step() along the way to update its observations

        task.step(action=NavigationAction) ->
            NavigationAction.step() ->
                task.step(action=Other SimulatorTaskAction)

        :param task: EmbodiedTask instance with required attributes
        :param goal_position: 2d position in the local/global map with range in [-1, 1]
        """
        self._check_task_attributes(task)

        self.global_semantic_map = task.observations["global_semantic_map"]
        self.current_map_pose: np.ndarray = task.observations[MapPoseSensor.uuid]

        self._extract_current_obstacle_and_explored_maps()
        self.previous_explored_map = self.current_explored_map.copy()

        # When local_semantic_map exists, the goal_position is specified with respect to
        # local map
        if "local_semantic_map" in task.observations:
            self.use_local_representation = True
            self.local_semantic_map = task.observations["local_semantic_map"]

        self._reset_internal_state(goal_position, task.current_episode)

        path = self._generate_path()
        if path is None:
            return task.observations  # type: ignore

        primitive_action_planner = PrimitiveActionPlanner(
            self.config.PRIMITIVE_ACTION_PLANNER, path
        )

        if self.visualization:
            self._reset_semantic_map_viewer(task, **kwargs)
            self._visualize(task.observations, path)

        cont_left_step_count = 0
        cont_right_step_count = 0
        while True:
            action = primitive_action_planner.step(self.current_map_pose)
            observations = task.step(action, task.current_episode)  # type: ignore
            self.global_semantic_map = task.observations["global_semantic_map"]
            if self.use_local_representation:
                self.local_semantic_map = task.observations["local_semantic_map"]

            if np.linalg.norm(
                self.current_map_pose[:2] - observations[MapPoseSensor.uuid][:2]
            ) < 0.5 and (
                self.current_map_pose[2:] @ observations[MapPoseSensor.uuid][2:]
            ) > np.cos(
                self.config.PRIMITIVE_ACTION_PLANNER.ANGLE_THRESHOLD / 2
            ):
                self.trapped_count += 1
                if self.trapped_count > self.config.TRAPPED_BREAK_THRESHOLD:
                    # Don't forget to add extra action count
                    # Otherwise, the agent will learn to deliberately trap itself
                    # to get few slack penalty
                    task.primitive_actions_in_last_navigation += (
                        self.config.MAX_PRIMITIVE_STEPS - self.primitive_step_count - 1
                    )
                    break
            else:
                self.trapped_count = 0

            if action["action"] == "MOVE_FORWARD":
                self.primitive_step_count += 1
                cont_right_step_count = 0
                cont_left_step_count = 0
            # count left step ground truth for PathComplexityAuxiliaryTask
            if action["action"] == "TURN_LEFT":
                if (
                    cont_left_step_count == self.config.COUNTED_STEP_THRESHOLD - 1
                    and self.primitive_step_count
                ):
                    self.left_step_count += 1
                cont_left_step_count += 1
                cont_right_step_count = 0
            # count right step ground truth for PathComplexityAuxiliaryTask
            if action["action"] == "TURN_RIGHT":
                if (
                    cont_right_step_count == self.config.COUNTED_STEP_THRESHOLD - 1
                    and self.primitive_step_count
                ):
                    self.right_step_count += 1
                cont_right_step_count += 1
                cont_left_step_count = 0
            if task._sim.previous_step_collided:  # type: ignore
                self._check_unseen_obstacle(action, observations[MapPoseSensor.uuid])
                self.collided_count += 1

            self.current_map_pose = observations[MapPoseSensor.uuid]

            if (
                not self.previous_explored_map[
                    int(self.current_map_pose[0]), int(self.current_map_pose[1])
                ]
                and action["action"] == "MOVE_FORWARD"
                and self.primitive_step_count - self.last_planning_step
                > self.config.PATH_PLANNING_INTERVAL
            ) or self.collided_count > self.config.COLLIDED_PLANNING_THRESHOLD:
                self._extract_current_obstacle_and_explored_maps()
                self._add_unseen_obstacle(task, observations)
                # task.replanning_times += 1
                path = self._generate_path()
                if path is None:
                    break
                primitive_action_planner.path = path
                self.previous_explored_map = self.current_explored_map.copy()
                self.collided_count = 0
                self.last_planning_step = self.primitive_step_count

            if self.visualization:
                self._visualize(observations, primitive_action_planner.path)
            if (
                action["action"] == "STOP"
                or self.primitive_step_count >= self.config.MAX_PRIMITIVE_STEPS
            ):
                break

        # Look around at destination
        self.look_around(task)

        self._add_unseen_obstacle(task, observations)

        observations["previous_goal_position"] = (
            self.goal_position / (self.global_semantic_map.shape[0] / 2.0) - 1.0
        )

        task.left_step_count = self.left_step_count
        task.right_step_count = self.right_step_count
        return observations

    def look_around(self, task):
        r"""Turn around to scan surroundings"""
        for _ in range(self.config.NUM_TURN_AROUND):
            observations = task.step({"action": "TURN_LEFT"}, task.current_episode)  # type: ignore
            if self.visualization and hasattr(self, "_semantic_map_viewer"):
                self._semantic_map_viewer.goal_position = None
                self._semantic_map_viewer.goal_mean = None
                self._semantic_map_viewer.goal_stddev = None
                self._semantic_map_viewer.path = []
                self._visualize(observations)

    @profiling_wrapper.RangeContext(
        "NavigationAction._extract_current_obstacle_and_explored_maps"
    )
    def _extract_current_obstacle_and_explored_maps(self):
        r"""Extract obstacle and exploration channels from semantic map,
        and transfer tensors into cpu if they reside in gpu
        """
        if self.use_local_representation:
            self.normalized_obstacle_map = self.normalize_obstacle_map(
                self.local_semantic_map[..., SemanticTopDownSensor.obstacle_channel],
                self.config.OBSTACLE_THRESHOLD,
            )
        else:
            self.normalized_obstacle_map = self.normalize_obstacle_map(
                self.global_semantic_map[..., SemanticTopDownSensor.obstacle_channel],
                self.config.OBSTACLE_THRESHOLD,
            )

        self.current_explored_map = self.global_semantic_map[
            ..., SemanticTopDownSensor.exploration_channel
        ]
        if isinstance(self.current_explored_map, torch.Tensor):
            self.current_explored_map = self.current_explored_map.cpu().numpy()

    def _reset_internal_state(self, goal_position: np.ndarray, episode: Episode):
        r"""Reset goal related internal states and some counters

        :param goal_position: 2d vector with range in [-1, 1]
        :param episode: current experiencing episode
        """
        if issubclass(goal_position.dtype.type, np.integer):
            self.goal_position = goal_position
        else:
            if self.use_local_representation:
                self.goal_position = self.convert_local_goal_position_to_map_coordinate(
                    goal_position,
                    self.global_semantic_map.shape[0],
                    self.current_map_pose[:2],
                    self.local_semantic_map.shape[0],
                )
            else:
                self.goal_position = (
                    self.convert_global_goal_position_to_map_coordinate(
                        goal_position,
                        self.global_semantic_map.shape[0],
                    )
                )
        self.primitive_step_count = 0
        self.collided_count = 0
        self.trapped_count = 0
        self.left_step_count = 0
        self.right_step_count = 0
        if self._current_episode_id != episode.episode_id:
            self._current_episode_id = episode.episode_id
            if self.config.USE_SIMULATOR_ORACLE:
                self.collided_position_counter = defaultdict(lambda: 0)
                self.collided_positions = []
            else:
                self.collided_poses = []

    def _reset_semantic_map_viewer(self, task: ScanningTask, **kwargs):
        r"""Reset SemanticMapViewer for following visualization, including
        a custom obstacle normalization function, current goal position,

        :param task: EmbodiedTask instance to get sensor suite
        :param goal_mean: optional kwargs to visualize actual network output mean
        :param goal_stddev: optional kwargs to visualize standard deviation of network output
        """
        if not hasattr(self, "_semantic_map_viewer"):
            if SemanticTopDownCudaSensor.uuid in task.sensor_suite.sensors:
                semantic_channel_categories = task.sensor_suite.sensors.get(
                    SemanticTopDownCudaSensor.uuid
                ).semantic_channel_categories  # type: ignore
            elif SemanticTopDownSensor.uuid in task.sensor_suite.sensors:
                semantic_channel_categories = task.sensor_suite.sensors.get(
                    SemanticTopDownSensor.uuid
                ).semantic_channel_categories  # type: ignore
            else:
                raise AssertionError(
                    f"{type(self)} requires a SemanticTopDownSensor to compute"
                )

            def normalize_obstacle_fn(obstacle_map: np.ndarray) -> np.ndarray:
                return self.normalize_obstacle_map(
                    obstacle_map, self.config.OBSTACLE_THRESHOLD
                ).numpy()

            self._semantic_map_viewer = SemanticMapViewer(
                task.observations,
                num_channels=self.global_semantic_map.shape[2],
                category_name_map=semantic_channel_categories,
                normalize_obstacle_fn=normalize_obstacle_fn,
            )
        self._semantic_map_viewer.goal_position = self.goal_position

        if "goal_mean" in kwargs:
            if self.use_local_representation:
                self._semantic_map_viewer.goal_mean = (
                    self.convert_local_goal_position_to_map_coordinate(
                        kwargs["goal_mean"],
                        self.global_semantic_map.shape[0],
                        self.current_map_pose[:2],
                        self.local_semantic_map.shape[0],
                    )
                )
            else:
                self._semantic_map_viewer.goal_mean = (
                    self.convert_global_goal_position_to_map_coordinate(
                        kwargs["goal_mean"],
                        self.global_semantic_map.shape[0],
                    )
                )
        if "goal_stddev" in kwargs:
            self._semantic_map_viewer.goal_stddev = kwargs["goal_stddev"]

    @staticmethod
    def convert_global_goal_position_to_map_coordinate(
        goal_position: np.ndarray,
        map_size_in_cells: int,
    ) -> np.ndarray:
        r"""Convert the global map goal position in range [-1, 1] to [0, map_size_in_cells - 1]

        :param goal_position: 2d vector with range in [-1, 1] that specify a goal in global map
        :param map_size_in_cells: how many cells are there in the global map
        :return: 2d vector with range in [0, map_size_in_cells - 1]
        """
        assert np.all((-1.0 <= goal_position) & (goal_position <= 1.0)), (
            "input goal_position in NavigationAction has to be in range [-1.0, 1.0] for float type,"
            "to specify values in range [0, map_size_in_cells] using integer type instead"
        )

        goal_position = (goal_position + 1.0) / 2.0
        return np.array(
            (
                goal_position[0] * (map_size_in_cells - 1),
                goal_position[1] * (map_size_in_cells - 1),
            )
        )

    @staticmethod
    def convert_local_goal_position_to_map_coordinate(
        goal_position: np.ndarray,
        map_size_in_cells: int,
        map_position: np.ndarray,
        local_map_size_in_cells: int,
    ) -> np.ndarray:
        r"""Convert the local map goal position in range [-1, 1] to [0, map_size_in_cells - 1]

        :param goal_position: 2d vector with range in [-1, 1] that specify a goal in local map
        :param map_size_in_cells: how many cells are there in the global map
        :param map_position: agent current position in map coordinate
        :param local_map_size_in_cells: how many cells are there in the local map
        :return: 2d vector with range in [0, map_size_in_cells - 1]
        """
        assert np.all((-1.0 <= goal_position) & (goal_position <= 1.0)), (
            "input goal_position in NavigationAction has to be in range [-1.0, 1.0] for float type,"
            "to specify values in range [0, map_size_in_cells] using integer type instead"
        )

        half_length = local_map_size_in_cells // 2
        # Note that: if current agent position is (400, 500),
        # this will map (-1.0, -1.0) to (200, 300) and (1.0, 1.0) to
        # (600, 700) instead of (599, 699), which may not be that important
        global_map_goal = np.array(map_position + goal_position * half_length)

        return np.clip(global_map_goal, a_min=0, a_max=map_size_in_cells - 1)

    def _convert_local_path_to_global_path(
        self, path: List[np.ndarray]
    ) -> List[np.ndarray]:
        r"""Convert path waypoints that computed in local obstacle map into points of global map

        :param path: computed path waypoints in local map
        :return: converted path waypoints in global map
        """
        map_position = self.current_map_pose[:2].astype(np.int32)
        half_length = self.local_semantic_map.shape[0] // 2
        map_size_in_cells = self.global_semantic_map.shape[0]
        return [
            np.clip(
                map_position + point - half_length, a_min=0, a_max=map_size_in_cells - 1
            )
            for point in path
        ]

    @profiling_wrapper.RangeContext("NavigationAction._generate_path")
    def _generate_path(self) -> Optional[List[np.ndarray]]:
        r"""Generate path using path_planner

        :return: list of 2d path waypoints or None if cannot find a path
        """
        if self.use_local_representation:
            start_point = (
                np.array(self.local_semantic_map.shape[:2], dtype=np.int32) // 2
            )
            end_point = np.clip(
                self.goal_position.astype(np.int32)
                - self.current_map_pose[:2].astype(np.int32)
                + start_point,
                a_min=0,
                a_max=self.local_semantic_map.shape[0] - 1,
            )
        else:
            start_point = self.current_map_pose[:2].astype(np.int32)
            end_point = self.goal_position.astype(np.int32)
        obstacle_map = (
            self.obstacle_inflation(
                self.normalized_obstacle_map.unsqueeze(0),
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        obstacle_map[start_point[0], start_point[1]] = 0.0
        obstacle_map[
            max(end_point[0] - 1, 0) : min(end_point[0] + 2, obstacle_map.shape[0] - 1),
            max(end_point[1] - 1, 0) : min(end_point[1] + 2, obstacle_map.shape[1] - 1),
        ] = 0.0
        result = self.path_planner.find(
            obstacle_map=obstacle_map,
            start_point=start_point,
            end_point=end_point,
            obstacle_cost=self.config.PATH_FINDER_OBSTACLE_COST,
            iteration_threshold=self.config.PATH_FINDER_ITERATION_THRESHOLD,
        )
        # We are surrounded by obstacles, try once more with original
        # (without inflation) obstacle map
        if len(result) == 1:
            obstacle_map = self.normalized_obstacle_map.cpu().numpy()
            obstacle_map[start_point[0], start_point[1]] = 0.0
            obstacle_map[
                max(end_point[0] - 1, 0) : min(
                    end_point[0] + 2, obstacle_map.shape[0] - 1
                ),
                max(end_point[1] - 1, 0) : min(
                    end_point[1] + 2, obstacle_map.shape[1] - 1
                ),
            ] = 0.0
            result = self.path_planner.find(
                obstacle_map=obstacle_map,
                start_point=start_point,
                end_point=end_point,
                obstacle_cost=self.config.PATH_FINDER_OBSTACLE_COST,
                iteration_threshold=self.config.PATH_FINDER_ITERATION_THRESHOLD,
            )

        if len(result) > 1:
            # Convert path waypoints back to global map
            if self.use_local_representation:
                result = self._convert_local_path_to_global_path(result)
                # When the agent moves around the goal may fall out of the local map
                # Add this back mainly for visualization purpose
                if np.any(result[-1] != self.goal_position):
                    result.append(self.goal_position)
            return result
        else:
            return None

    def _visualize(
        self,
        observations: Observations,
        path: Optional[List[np.ndarray]] = None,
    ):
        r"""Visualize current obstacle map with agent pose

        :param observations: current agent's observations
        :param path: path found by path finder that to be drawn on image
        """
        self._semantic_map_viewer.update_observations(observations)
        if path is not None:
            self._semantic_map_viewer.path = path
        self._semantic_map_viewer.update_figure()
        plt.pause(0.02)

    def _check_unseen_obstacle(self, action: Dict[str, Any], map_pose: np.ndarray):
        r"""Check if there is an unseen obstacle before the agent, and add it to the candidate list
        for future processing

        :param action: current action adopted by the agent
        :param map_pose: agent pose when action finished, self.current_map_pose is actually previous
            map pose here, since we haven't update it yet
        """
        if self.config.USE_SIMULATOR_ORACLE:
            collided_position = tuple(map_pose[:2].astype(np.int32))
            self.collided_position_counter[collided_position] += 1
            if (
                self.collided_position_counter[collided_position]
                >= self.config.UNSEEN_OBSTACLE_ADDED_THRESHOLD
            ):
                self.collided_positions.append(collided_position)
                del self.collided_position_counter[collided_position]
        elif action["action"] == "MOVE_FORWARD":
            move_distance = np.linalg.norm(map_pose[:2] - self.current_map_pose[:2])
            if move_distance < 0.5:
                self.collided_poses.append(map_pose)
            else:
                # Use previous head direction dot actual move direction to measure
                # whether the agent was collided or not, since the agent may slide
                # before the obstacle
                actual_move_direction = (
                    map_pose[:2] - self.current_map_pose[:2]
                ) / move_distance
                if (self.current_map_pose[2:] @ actual_move_direction) < np.cos(
                    np.pi / 4
                ):
                    self.collided_poses.append(map_pose)

    @profiling_wrapper.RangeContext("NavigationAction._add_unseen_obstacle")
    def _add_unseen_obstacle(self, task: EmbodiedTask, observations: Observations):
        r"""Add some unobservable obstacles or scene deficiencies to observations by looking up
        simulator is_navigable oracle or compute obstacle cells from agent's head direction when
        collision

        :param task: EmbodiedTask that can access Simulator
        :param observations: current agent's observations
        """
        if self.config.USE_SIMULATOR_ORACLE:
            if len(self.collided_positions) == 0:
                return
        else:
            if len(self.collided_poses) == 0:
                return

        obstacle_map = observations["global_semantic_map"][
            ..., SemanticTopDownSensor.obstacle_channel
        ]

        if isinstance(obstacle_map, torch.Tensor):
            if not hasattr(self, "original_obstacle_map"):
                self.original_obstacle_map = torch.empty_like(
                    obstacle_map, device="cpu"
                ).pin_memory()
            self.original_obstacle_map.copy_(obstacle_map)
            obstacle_map = self.original_obstacle_map.numpy()

        if self.config.USE_SIMULATOR_ORACLE:
            self._add_unseen_obstacle_by_simulator_oracle(task, obstacle_map)
        else:
            self._add_unseen_obstacle_by_head_direction(obstacle_map)

        if isinstance(observations["global_semantic_map"], torch.Tensor):
            observations["global_semantic_map"][
                ..., SemanticTopDownSensor.obstacle_channel
            ] = self.original_obstacle_map.cuda(
                observations["global_semantic_map"].device
            )

    @profiling_wrapper.RangeContext(
        "NavigationAction._add_unseen_obstacle_by_simulator_oracle"
    )
    def _add_unseen_obstacle_by_simulator_oracle(self, task, obstacle_map):
        r"""Add some unobservable obstacles to obstacle map by querying the simulator for
        navigability

        :param task: EmbodiedTask that can access Simulator
        :param obstacle_map: obstacle map to be changed
        """
        start_index = -(self.config.CHECK_UNSEEN_OBSTACLES_RANGE // 2)
        end_index = self.config.CHECK_UNSEEN_OBSTACLES_RANGE // 2 + 1

        for collided_position in self.collided_positions:
            for i in range(start_index, end_index):
                for j in range(start_index, end_index):
                    neighbor_position = np.array(
                        [collided_position[0] + i, collided_position[1] + j]
                    )
                    if (
                        np.any(neighbor_position < 0)
                        or np.any(neighbor_position >= obstacle_map.shape[0])
                        or obstacle_map[neighbor_position[0], neighbor_position[1]]
                        >= self.config.OBSTACLE_THRESHOLD - 1e-6
                    ):
                        continue
                    if not self._is_navigable_in_map_coordinate(
                        task, neighbor_position
                    ):
                        obstacle_map[
                            neighbor_position[0], neighbor_position[1]
                        ] = self.config.OBSTACLE_THRESHOLD

        self.collided_positions = []

    def _add_unseen_obstacle_by_head_direction(self, obstacle_map):
        r"""Add some unobservable obstacles to obstacle map by checking agent's head direction when
        collision

        :param obstacle_map: obstacle map to be changed
        """
        for collided_pose in self.collided_poses:
            obstacle_tangent = np.array([-collided_pose[3], collided_pose[2]])
            for i in range(2, 4):
                obstacle_position = collided_pose[:2] + i * collided_pose[2:]
                for j in range(-3, 4):
                    neighbor_position = (
                        obstacle_position + j * obstacle_tangent
                    ).astype(np.int32)
                    np.clip(
                        neighbor_position,
                        a_min=0,
                        a_max=obstacle_map.shape[0] - 1,
                        out=neighbor_position,
                    )
                    obstacle_map[
                        neighbor_position[0], neighbor_position[1]
                    ] = self.config.OBSTACLE_THRESHOLD

        self.collided_poses = []

    def _is_navigable_in_map_coordinate(
        self, task: EmbodiedTask, query_point: np.ndarray
    ) -> bool:
        r"""Check whether a point in map coordinate is navigable according to
        simulator.is_navigable() method

        :param task: EmbodiedTask that can access Simulator
        :param query_point: 2d point in map coordinate
        :return: true if the query point is navigable
        """
        world_point = task.sensor_suite.sensors.get(
            MapPoseSensor.uuid
        ).convert_map_coordinate_to_world_coordinate(  # type: ignore
            query_point
        )

        return task._sim.is_navigable(world_point)  # type: ignore

    @staticmethod
    def _check_task_attributes(task):
        r"""Check if the EmbodiedTask instance has observations and current_episode
        attributes

        :param task: EmbodiedTask instance to check
        """
        assert hasattr(
            task, "observations"
        ), "NavigationAction only works with EmbodiedTask that has saved latest observations"

        assert hasattr(
            task, "current_episode"
        ), "NavigationAction only works with EmbodiedTask that has saved current episode"

    def _initialize_obstacle_inflation(self):
        r"""Inflate obstacle cells to their neighbors to make the agent prefer safer path"""

        def obstacle_inflation_impl(obstacle_map):
            averaged_obstacle_map = F.avg_pool2d(
                obstacle_map, kernel_size=9, stride=1, padding=4
            )
            return torch.max(obstacle_map, averaged_obstacle_map)

        self.obstacle_inflation = obstacle_inflation_impl

    @staticmethod
    def normalize_obstacle_map(
        raw_obstacle_map: Union[np.ndarray, torch.Tensor], obstacle_threshold: int
    ) -> torch.Tensor:
        r"""Convert raw obstacle map into [0, 1] value range

        :param raw_obstacle_map: obstacle map from SemanticMapBuilder
        :param obstacle_threshold: the point number threshold that deemed can't be pass through
        """
        result = (raw_obstacle_map / obstacle_threshold) ** 2
        if isinstance(result, torch.Tensor):
            return torch.clip(result, min=0.0, max=1.0)
        else:
            return torch.from_numpy(np.clip(result, a_min=0.0, a_max=1.0))
