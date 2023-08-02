#  ____                  _           _
# / ___|  ___ __ _ _ __ | |__   ___ | |_
# \___ \ / __/ _` | '_ \| '_ \ / _ \| __|
#  ___) | (_| (_| | | | | |_) | (_) | |_
# |____/ \___\__,_|_| |_|_.__/ \___/ \__|

"""
    habitat_scanbot2d.measures
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Measure classes for scanning task

    :copyright: (c) 2021 by XiXia.
    :license: MIT License, see LICENSE for more details.
"""

from collections import defaultdict
from typing import Any, Optional, Dict, cast, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from habitat.config.default import Config
from habitat.core.registry import registry
from habitat.core.dataset import Episode
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.utils.visualizations import maps
from habitat_scanbot2d.utils.visualization import visualize_single_channel_image

from habitat_scanbot2d.sensors import SemanticTopDownSensor, MapPoseSensor
from habitat_scanbot2d.cuda_sensors import SemanticTopDownCudaSensor
from habitat_scanbot2d.navigation_action import NavigationAction


@registry.register_measure
class CompletedArea(Measure):
    r"""Measure how many areas has been explored by the agent

    This measure depends on SemanticTopDownSensor.
    """

    uuid: str = "completed_area"

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        self._config = config
        self.map_cell_size: float
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def reset_metric(
        self, *args: Any, task: EmbodiedTask, observations: Observations, **kwargs: Any
    ) -> None:
        if SemanticTopDownCudaSensor.uuid in task.sensor_suite.sensors:
            self.map_cell_size = task.sensor_suite.sensors.get(
                SemanticTopDownCudaSensor.uuid
            ).map_cell_size  # type: ignore
        elif SemanticTopDownSensor.uuid in task.sensor_suite.sensors:
            self.map_cell_size = task.sensor_suite.sensors.get(
                SemanticTopDownSensor.uuid
            ).map_cell_size  # type: ignore
        else:
            raise AssertionError(
                f"{type(self)} requires a SemanticTopDownSensor to compute"
            )

        self.update_metric(*args, observations=observations, **kwargs)

    def update_metric(self, *args: Any, observations: Observations, **kwargs: Any):
        explored_map = observations["global_semantic_map"][
            ..., SemanticTopDownSensor.exploration_channel
        ]
        if isinstance(explored_map, torch.Tensor):
            self._metric = torch.sum(explored_map).item() * (self.map_cell_size ** 2)  # type: ignore
        else:
            self._metric = np.sum(explored_map) * (self.map_cell_size ** 2)


@registry.register_measure
class CompletedRate(Measure):
    r"""Measure how many percentage of the total area has been explored by the agent

    This measure depends on CompletedArea measure.
    """

    uuid: str = "completed_rate"

    def __init__(
        self, sim: HabitatSim, config: Config, *args: Any, **kwargs: Any
    ) -> None:
        self._sim = sim
        self._config = config
        self.map_cell_size: float
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _is_on_same_floor(
        self,
        height: float,
        floor_height: Optional[float] = None,
        ceiling_height: float = 2.0,
    ) -> bool:
        if floor_height is None:
            floor_height = self._sim.get_agent_state().position[1]
        return floor_height < height < floor_height + ceiling_height  # type: ignore

    def _add_objects_occupied_area(self) -> None:
        r"""Add area occupied by objects that in the same floor into the topdown map"""
        objects = self._sim.semantic_annotations().objects
        for obj in objects:
            if obj is not None:
                center = obj.aabb.center
                x_len, _, z_len = obj.aabb.sizes / 2.0
                corners = [
                    center + np.array([x, 0, z])
                    for x, z in [
                        (-x_len, -z_len),
                        (x_len, z_len),
                    ]
                    if self._is_on_same_floor(center[1])
                ]

                if len(corners):
                    (x_min, y_min), (x_max, y_max) = [
                        maps.to_grid(
                            point[2],
                            point[0],
                            self.entire_topdown_map.shape[:2],  # type: ignore
                            sim=self._sim,
                        )
                        for point in corners
                    ]

                    self.entire_topdown_map[
                        x_min : x_max + 1, y_min : y_max + 1
                    ] = maps.MAP_VALID_POINT

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [CompletedArea.uuid])
        self.map_cell_size = task.measurements.measures[
            CompletedArea.uuid
        ].map_cell_size  # type: ignore

        # topdown_map: image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        # the flag is set).
        self.entire_topdown_map = maps.get_topdown_map_from_sim(
            self._sim,
            draw_border=False,
            meters_per_pixel=self.map_cell_size,
        ).astype(np.float32)
        empty_area = np.sum(self.entire_topdown_map) * (self.map_cell_size ** 2)
        self._add_objects_occupied_area()
        self.object_occupied_area = (
            np.sum(self.entire_topdown_map) * (self.map_cell_size ** 2) - empty_area
        )
        # inflate the valid space a little bit, since the walls may be considered as explored
        # in semantic_topdown sensor
        self.entire_topdown_map = (
            F.max_pool2d(
                torch.from_numpy(self.entire_topdown_map).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1,
            )
            .numpy()
            .squeeze()
        )
        self.entire_space_area = np.sum(self.entire_topdown_map) * (
            self.map_cell_size ** 2
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        completed_area = task.measurements.measures[CompletedArea.uuid].get_metric()
        self._metric = completed_area / self.entire_space_area


@registry.register_measure
class ScanningSuccess(Measure):
    r"""Whether or not the agent successfully scanned and reconstructed the scene

    This measure depends on CompletedRate measure.
    """

    uuid: str = "scanning_success"

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        self._config = config
        self._achieved_rate_statistics = {}
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def reset_metric(
        self, *args: Any, episode: Episode, task: EmbodiedTask, **kwargs: Any
    ) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [CompletedRate.uuid])
        self.existing_scanning_rate = ScanningRate.uuid in task.measurements.measures
        if self.existing_scanning_rate:
            task.measurements.check_measure_dependencies(self.uuid, [ScanningRate.uuid])

        self._completed_rate_threshold = self._config.SUCCESS_COMPLETED_RATE
        self._scanning_rate_threshold = self._config.SUCCESS_SCANNING_RATE

        if self._config.USE_ADAPTIVE_RATE:
            self.scene_name = Path(episode.scene_id).stem
            self.current_region_id, belonging_score = self.find_belonging_region(
                cast(HabitatSim, task._sim)
            )

            if (
                self.scene_name in self._achieved_rate_statistics
                and self.current_region_id
                in self._achieved_rate_statistics[self.scene_name]
            ):
                region_stat = self._achieved_rate_statistics[self.scene_name][
                    self.current_region_id
                ]
                self._completed_rate_threshold = (
                    1 - self._config.ADAPTIVE_COEFFICIENT
                ) * region_stat[
                    "average_completed_rate"
                ] + self._config.ADAPTIVE_COEFFICIENT * region_stat[
                    "max_completed_rate"
                ]

                if self.existing_scanning_rate:
                    self._scanning_rate_threshold = (
                        1 - self._config.ADAPTIVE_COEFFICIENT
                    ) * region_stat[
                        "average_scanning_rate"
                    ] + self._config.ADAPTIVE_COEFFICIENT * region_stat[
                        "max_scanning_rate"
                    ]

        self.update_metric(task=task)

    @staticmethod
    def find_belonging_region(sim: HabitatSim) -> Tuple[str, float]:
        current_position = sim.get_agent_state().position
        best_belonging_score = np.inf
        best_region_id = ""
        for region in sim.semantic_annotations().regions:
            position_diff = np.abs(current_position - region.aabb.center)
            belonging_score = np.max(position_diff - region.aabb.sizes / 2.0)
            if belonging_score < best_belonging_score:
                best_belonging_score = belonging_score
                best_region_id = region.id

        return best_region_id, best_belonging_score

    def update_statistics(self, completed_rate, scanning_rate: Optional[float]):
        if self._config.USE_ADAPTIVE_RATE:
            if self.scene_name not in self._achieved_rate_statistics:
                self._achieved_rate_statistics[self.scene_name] = {}
            if (
                self.current_region_id
                not in self._achieved_rate_statistics[self.scene_name]
            ):
                self._achieved_rate_statistics[self.scene_name][self.current_region_id] = {}

            region_stat = self._achieved_rate_statistics[self.scene_name][
                self.current_region_id
            ]
            if len(region_stat) == 0:
                region_stat["average_completed_rate"] = completed_rate
                region_stat["max_completed_rate"] = completed_rate

                if self.existing_scanning_rate:
                    region_stat["average_scanning_rate"] = scanning_rate
                    region_stat["max_scanning_rate"] = scanning_rate

            else:
                region_stat["average_completed_rate"] = (
                    1 - self._config.AVERAGE_COEFFICIENT
                ) * region_stat[
                    "average_completed_rate"
                ] + self._config.AVERAGE_COEFFICIENT * completed_rate

                region_stat["max_completed_rate"] = max(
                    region_stat["max_completed_rate"],
                    completed_rate,
                )

                if self.existing_scanning_rate and scanning_rate is not None:
                    region_stat["average_scanning_rate"] = (
                        1 - self._config.AVERAGE_COEFFICIENT
                    ) * region_stat[
                        "average_scanning_rate"
                    ] + self._config.AVERAGE_COEFFICIENT * scanning_rate

                    region_stat["max_scanning_rate"] = max(
                        region_stat["max_scanning_rate"],
                        scanning_rate,
                    )

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        completed_rate = task.measurements.measures[CompletedRate.uuid].get_metric()
        if self.existing_scanning_rate:
            scanning_rate = task.measurements.measures[ScanningRate.uuid].get_metric()
        else:
            scanning_rate = 100.0

        if (
            completed_rate > self._completed_rate_threshold
            and scanning_rate > self._scanning_rate_threshold
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0

    @property
    def achieved_rate_statistics(self):
        result = {}
        for scene_name, regions in self._achieved_rate_statistics.items():
            result[scene_name] = {}
            result[scene_name]["max_completed_rate"] = max(
                region_stat["max_completed_rate"] for region_stat in regions.values()
            )
            result[scene_name]["max_scanning_rate"] = max(
                region_stat["max_scanning_rate"] for region_stat in regions.values()
            )
            result[scene_name]["min_completed_rate"] = min(
                region_stat["max_completed_rate"] for region_stat in regions.values()
            )
            result[scene_name]["min_scanning_rate"] = min(
                region_stat["max_scanning_rate"] for region_stat in regions.values()
            )
            result[scene_name]["average_completed_rate"] = np.mean(
                [
                    region_stat["average_completed_rate"]
                    for region_stat in regions.values()
                ]
            )
            result[scene_name]["average_scanning_rate"] = np.mean(
                [
                    region_stat["average_scanning_rate"]
                    for region_stat in regions.values()
                ]
            )
        return result


@registry.register_measure
class LongTermGoalReachability(Measure):
    r"""Measure wether the given long term goal is reachable and return
    the final difference between the agent actually get

    This measure depends on MapPoseSensor.
    """

    uuid: str = "long_term_goal_reachability"

    def __init__(
        self, sim: HabitatSim, config: Config, *args: Any, **kwargs: Any
    ) -> None:
        self._sim = sim
        self._config = config
        self.map_cell_size: float
        self.map_size_in_cells: int
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        assert (
            MapPoseSensor.uuid in task.sensor_suite.sensors
        ), f"{type(self)} requires a MapPoseSensor to compute"
        map_pose_sensor = task.sensor_suite.sensors.get(MapPoseSensor.uuid)
        self.map_cell_size = map_pose_sensor.map_cell_size  # type: ignore
        self.map_size_in_cells = map_pose_sensor.map_size_in_cells  # type: ignore
        self._metric = 0.0

    def update_metric(
        self,
        *args: Any,
        action: Dict[str, Any],
        observations: Observations,
        **kwargs: Any,
    ) -> None:
        current_pose = observations[MapPoseSensor.uuid]
        if "previous_goal_position" in observations:
            goal_position = (
                (observations["previous_goal_position"] + 1.0)
                * self.map_size_in_cells / 2.0
            )
        else:
            goal_position = (
                NavigationAction.convert_global_goal_position_to_map_coordinate(
                    action["action_args"]["goal_position"],
                    self.map_size_in_cells,
                )
            )
        self._metric += (
            np.linalg.norm(goal_position - current_pose[:2]) * self.map_cell_size
        )


@registry.register_measure
class PrimitiveActionCount(Measure):
    r"""Count how many primitive actions have been taken during this episode,
    which is helpful to compute slack reward when using navigation action
    """

    uuid: str = "primitive_action_count"

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        assert hasattr(
            task, "primitive_actions_in_last_navigation"
        ), f"{type(self)} requires an EmbodiedTask instance with primitive_actions_in_last_navigation attribute"
        self._metric = 0

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        self._metric += task.primitive_actions_in_last_navigation  # type: ignore


@registry.register_measure
class ScanningQuality(CompletedArea):
    r"""Measure how many quality related scores has been achieved by the agent

    This measure depends on SemanticTopDownSensor.
    """

    uuid: str = "scanning_quality"

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, config=config, **kwargs)

    def reset_metric(
        self, *args: Any, task: EmbodiedTask, observations: Observations, **kwargs: Any
    ) -> None:
        if SemanticTopDownCudaSensor.uuid in task.sensor_suite.sensors:
            self.num_quality_channels = task.sensor_suite.sensors.get(
                SemanticTopDownCudaSensor.uuid
            ).num_quality_channels  # type: ignore
        elif SemanticTopDownSensor.uuid in task.sensor_suite.sensors:
            self.num_quality_channels = task.sensor_suite.sensors.get(
                SemanticTopDownSensor.uuid
            ).num_quality_channels  # type: ignore

        self._metric = 0.0
        super().reset_metric(*args, task=task, observations=observations, **kwargs)

    def update_metric(self, *args: Any, observations: Observations, **kwargs: Any):
        if self.num_quality_channels > 0:
            scanning_quality_map = observations["global_semantic_map"][
                ...,
                SemanticTopDownSensor.quality_channel : SemanticTopDownSensor.quality_channel
                + self.num_quality_channels,
            ]
            if isinstance(scanning_quality_map, torch.Tensor):
                self._metric = torch.sum(scanning_quality_map).item() * (  # type: ignore
                    self.map_cell_size ** 2
                )
            else:
                self._metric = np.sum(scanning_quality_map) * (self.map_cell_size ** 2)


@registry.register_measure
class ScanningRate(Measure):
    r"""Measure how many percentage of the objects has been scanned by the agent

    This measure depends on ScanningQuality measure.
    """

    uuid: str = "scanning_rate"

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [CompletedRate.uuid])
        task.measurements.check_measure_dependencies(self.uuid, [ScanningQuality.uuid])
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        self._metric = (
            task.measurements.measures[ScanningQuality.uuid].get_metric()
            / task.measurements.measures[CompletedRate.uuid].object_occupied_area  # type: ignore
        )


@registry.register_measure
class LeftStepCount(Measure):
    r"""Measure how many left primitive actions had been taken in a navigation action"""

    uuid: str = "left_step_count"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._metric = 0

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        if hasattr(task, "left_step_count"):
            self._metric = task.left_step_count  # type: ignore
        else:
            self._metric = 0


@registry.register_measure
class RightStepCount(Measure):
    r"""Measure how many right primitive actions had been taken in a navigation action"""

    uuid: str = "right_step_count"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._metric = 0

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        if hasattr(task, "right_step_count"):
            self._metric = task.right_step_count  # type: ignore
        else:
            self._metric = 0


@registry.register_measure
class QualityIncreaseRatio(Measure):
    r"""Measure how many map grids in quality channels of SemanticTopDownSensor
    had been increased in a navigation action"""

    uuid: str = "quality_increase_ratio"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        self._metric = 0.0

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        if hasattr(task, "quality_increase_ratio"):
            self._metric = task.quality_increase_ratio # type: ignore
        else:
            self._metric = 0.0

@registry.register_measure
class ObjectDiscovery(Measure):

    uuid: str = "object_discovery"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._instance_label_count = defaultdict(lambda: 0)
        self._num_objects = 1

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        self._instance_label_count = defaultdict(lambda: 0)
        self._metric = 0
        self._num_objects = 0

        objects = task._sim.semantic_annotations().objects # type: ignore
        floor_height = task._sim.get_agent_state().position[1]
        for obj in objects:
            if obj is not None:
                center = obj.aabb.center
                if floor_height < center[1] < floor_height + 3.0:
                    self._num_objects += 1

    def update_metric(self, *args: Any, observations: Observations, **kwargs: Any):
        semantic_labels = observations["semantic"][(observations["depth"] < 3.0).squeeze()]
        if isinstance(semantic_labels, torch.Tensor):
            unique_labels = torch.unique(semantic_labels).cpu().numpy()
        else:
            unique_labels = np.unique(semantic_labels)
        for label in unique_labels:
            self._instance_label_count[label] += 1

        self._metric = 0
        for count in self._instance_label_count.values():
            if count > 10:
                self._metric += 1
        self._metric /=  self._num_objects

@registry.register_measure
class ObjectCoverage(Measure):

    uuid: str = "object_coverage"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def reset_metric(
        self, *args: Any, task: EmbodiedTask, observations: Observations, **kwargs: Any
    ) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [CompletedRate.uuid])
        if SemanticTopDownCudaSensor.uuid in task.sensor_suite.sensors:
            self.num_quality_channels = task.sensor_suite.sensors.get(
                SemanticTopDownCudaSensor.uuid
            ).num_quality_channels  # type: ignore
        elif SemanticTopDownSensor.uuid in task.sensor_suite.sensors:
            self.num_quality_channels = task.sensor_suite.sensors.get(
                SemanticTopDownSensor.uuid
            ).num_quality_channels  # type: ignore
        self.map_cell_size = task.measurements.measures[
            CompletedRate.uuid
        ].map_cell_size  # type: ignore

        self.update_metric(*args, task=task, observations=observations, **kwargs)

    def update_metric(self, *args: Any, task: EmbodiedTask, observations: Observations, **kwargs: Any):
        scanning_quality_map = observations["global_semantic_map"][
            ...,
            SemanticTopDownSensor.quality_channel : SemanticTopDownSensor.quality_channel
            + self.num_quality_channels,
        ]
        if isinstance(scanning_quality_map, torch.Tensor):
            self._metric = torch.sum(torch.sum(scanning_quality_map, dim=2) > 0.0).item() * (  # type: ignore
                self.map_cell_size ** 2
            )
        else:
            self._metric = np.sum(np.sum(scanning_quality_map, axis=2) > 0.0) * (self.map_cell_size ** 2)

        self._metric /= task.measurements.measures[CompletedRate.uuid].object_occupied_area  # type: ignore
