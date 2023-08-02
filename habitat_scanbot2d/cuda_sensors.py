#  ____                  _           _
# / ___|  ___ __ _ _ __ | |__   ___ | |_
# \___ \ / __/ _` | '_ \| '_ \ / _ \| __|
#  ___) | (_| (_| | | | | |_) | (_) | |_
# |____/ \___\__,_|_| |_|_.__/ \___/ \__|

"""
    habitat_scanbot2d.sensors_cuda
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Definition of scanning task related Sensors using torch.Tensor in cuda device

    :copyright: (c) 2021 by XiXia.
    :license: MIT License, see LICENSE for more details.
"""

from typing import Any, Dict, cast
import numpy as np
import torch
import open3d as o3d

from gym import spaces, Space
from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, Simulator, SensorTypes
from habitat.utils import profiling_wrapper
from habitat_scanbot2d.utils.semantic_map import construct_transformation_matrix
from habitat_scanbot2d.sensors import (
    PointCloudSensor,
    CategorySemanticSensor,
    SemanticTopDownSensor,
)
from habitat_scanbot2d.channels_constructor import (
    construct_all_channels,
    construct_quality_and_semantic_channels,
)


@registry.register_sensor(name="PointCloudCudaSensor")
class PointCloudCudaSensor(PointCloudSensor):
    r"""Project agents current depth and image into a 3d point cloud with observations
    input as cuda torch.Tensor

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointCloud sensor.
    """
    uuid = "point_cloud_cuda"

    def __init__(self, sim: Simulator, config: Config, **kwargs):
        super().__init__(sim=sim, config=config)

    @profiling_wrapper.RangeContext("PointCloudSensor._compute_point_cloud")
    def _compute_point_cloud(self, depth: torch.Tensor) -> torch.Tensor:
        r"""Return point cloud in camera coordinate

        :param depth: depth observation with [height, width, 1]
        :return: xyz 3d points
        """
        assert depth.shape[:2] == (
            self._input_height,
            self._input_width,
        ), f"Incompatible depth image size {depth.shape[:2]}"
        fx = 1 / np.tan(self._input_hfov / 2.0)
        fy = self._input_width / self._input_height * fx

        with torch.cuda.device(depth.device):
            # [-1, 1] for x and [1, -1] for y as image coordinate is y-down while world is y-up
            v_grid, u_grid = torch.meshgrid(
                torch.linspace(1, -1, self._input_height, device="cuda"),
                torch.linspace(-1, 1, self._input_width, device="cuda"),
            )
            flatten_depth = depth.reshape(-1, 1)

        # K(x, y, z, 1)^T = (uz, vz, z, 1)^T
        # negate depth as the camera looks along -Z
        return torch.hstack(
            (
                flatten_depth * u_grid.reshape(-1, 1) / fx,
                flatten_depth * v_grid.reshape(-1, 1) / fy,
                -flatten_depth,
            )
        )


@registry.register_sensor(name="CategorySemanticCudaSensor")
class CategorySemanticCudaSensor(CategorySemanticSensor):
    r"""Provide semantic labels corresponding to object categories instead of object instances
    with observations input as cuda torch.Tensor

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the CategorySemantic sensor.
    """
    uuid: str = "category_semantic_cuda"

    def __init__(self, sim: Simulator, config: Config, **kwargs):
        super().__init__(sim=sim, config=config)
        self.category_mapping: torch.Tensor

    def update_instance_to_category_mapping(
        self, episode: Episode, device: torch.device
    ):
        r"""Build a mapping from instance label to category label in current scene"""
        if self._current_scene_id != episode.scene_id:
            self._current_scene_id = episode.scene_id

            scene = self._sim.semantic_annotations()
            instance_id_to_category_id = {
                int(obj.id.split("_")[-1]): obj.category.index()
                for obj in scene.objects
                if obj is not None
            }
            first_instance_label = min(instance_id_to_category_id.keys())
            # Make 0 label remains 0, if instance label start from 1
            self.category_mapping = torch.tensor(
                [0] * first_instance_label
                + [
                    instance_id_to_category_id[
                        i + first_instance_label
                    ]  # instance id may start from 1
                    for i in range(len(instance_id_to_category_id))
                ],
                dtype=torch.int32,
                device=device,
            )

    @profiling_wrapper.RangeContext("CategorySemanticSensor.get_observation")
    def get_observation(
        self, *args: Any, observations: Dict[str, Any], episode: Episode, **kwargs: Any
    ) -> torch.Tensor:
        self.update_instance_to_category_mapping(
            episode, observations["semantic"].device
        )
        assert torch.min(observations["semantic"]).item() >= 0
        return torch.take(self.category_mapping, observations["semantic"].long())


@registry.register_sensor(name="SemanticTopDownCudaSensor")
class SemanticTopDownCudaSensor(SemanticTopDownSensor):
    r"""Project agents current depth and semantic image into a topdown manner, this sensor will
        use PointCloudCudaSensor and CategorySemanticCudaSensor internally

    Args:
        sim: reference to the simulator for calculating task observations
        config: config for the SemanticTopDown sensor
    """
    uuid: str = "semantic_topdown_cuda"

    def __init__(self, sim: Simulator, config: Config, **kwargs):
        super().__init__(sim, config)
        # Use cuda version of these two sensors
        self._point_cloud_sensor = PointCloudCudaSensor(sim, config.POINT_CLOUD_SENSOR)
        self._category_semantic_sensor = CategorySemanticCudaSensor(
            sim, config.CATEGORY_SEMANTIC_SENSOR
        )
        self.current_pose: torch.Tensor
        self.use_torch_extensions = config.USE_TORCH_EXTENSIONS

    def _transform_local_to_global(self, point_cloud: torch.Tensor) -> torch.Tensor:
        r"""Transform point cloud in camera coordinate to global coordinate

        :param point_cloud: point cloud in camera coordinate, **note that** input point cloud may
            contain not only xyz, so we will only change the first three channels
        :return: transformed point cloud
        """
        self.current_pose = self.current_pose.to(point_cloud.device)
        local_xyz = point_cloud[:, :3]
        homogeneous_xyz = torch.hstack((local_xyz, torch.ones_like(local_xyz[:, :1])))
        point_cloud[:, :3] = (homogeneous_xyz @ self.current_pose.T)[:, :3]
        return point_cloud

    def _filter_too_far_or_too_close(self, point_cloud: torch.Tensor) -> torch.Tensor:
        r"""Remove points that are too far or too close to the camera

        :param point_cloud: point cloud in camera coordinate
        :return: filtered point cloud
        """
        medium_range_indexes = (torch.abs(point_cloud[:, 2]) < self.far_threshold) & (
            torch.abs(point_cloud[:, 2]) > self.near_threshold
        )
        return point_cloud[medium_range_indexes]

    def _transform_global_to_map(self, projected_points: torch.Tensor) -> torch.Tensor:
        r"""Projected 2d points in global coordinate into map coordinate by applying a scaling
        and a translation transformation

        :param projected_points: projected 2d points without y values in global coordinate
        :return: HxWxC array with first two channels represent (row, col)
        """
        translation = self.map_size_in_cells // 2
        projected_points[:, :2] = (
            projected_points[:, :2] / self.map_cell_size + translation
        )
        result = projected_points.long()
        torch.clip(
            result[:, :2], min=0, max=self.map_size_in_cells - 1, out=result[:, :2]
        )
        return result

    @profiling_wrapper.RangeContext("SemanticTopDownSensor._count_points_in_each_cell")
    def _count_points_in_each_cell(self, map_points: torch.Tensor) -> torch.Tensor:
        r"""Count how many points belong to a individual cell

        :param map_points: projected 2d points in map coordinate
        :param topdown_map: HxW array to be filled, echo cell represents the count of
            points falling into it
        """
        flattened_indexes = map_points[:, 0] * self.map_size_in_cells + map_points[:, 1]
        point_count_map = torch.bincount(
            flattened_indexes, minlength=self.map_size_in_cells ** 2
        )
        return point_count_map.reshape(
            self.map_size_in_cells, self.map_size_in_cells
        ).float()

    @profiling_wrapper.RangeContext(
        "SemanticTopDownSensor._construct_semantic_and_quality_channels"
    )
    def _construct_quality_and_semantic_channels(
        self, semantic_map_points: torch.Tensor
    ):
        agent_position = np.array(
            [self.current_pose[2, 3].item(), self.current_pose[0, 3].item()],
            dtype=np.int32,
        )
        agent_position = (
            super()._transform_global_to_map(agent_position[np.newaxis, :]).squeeze()
        )

        num_semantic_channels = self.num_total_channels - self.num_quality_channels - 3
        assert num_semantic_channels == len(
            self.semantic_channel_categories
        ), "NUM_TOTAL_CHANNELS mismatches with SEMANTIC_CHANNEL_CATEGORIES"
        semantic_channel_begin = self.num_quality_channels + 2

        need_other_categories = "others" in self.semantic_channel_categories

        if not isinstance(self.semantic_channel_labels, torch.Tensor):
            self.semantic_channel_labels = torch.tensor(
                self.semantic_channel_labels,
                dtype=torch.long,
                device=semantic_map_points.device,
            )

        # Since we hard-coded MAX_NUM_INSTANCES = 4096 in kernel file
        assert len(self._category_semantic_sensor.category_mapping) <= 4096

        construct_quality_and_semantic_channels(
            semantic_map_points,
            self.topdown_map,
            list(agent_position),
            self.quality_channel,
            self.map_cell_size,
            self.far_threshold,
            self.best_scanning_distance,
            self.orientation_division_interval,
            self.semantic_channel_labels,
            self._category_semantic_sensor.category_mapping,
            semantic_channel_begin,
            need_other_categories,
        )

    @profiling_wrapper.RangeContext("SemanticTopDownSensor._construct_all_channels")
    def _construct_all_channels(self, semantic_local_points: torch.Tensor):
        num_semantic_channels = self.num_total_channels - self.num_quality_channels - 3
        assert num_semantic_channels == len(
            self.semantic_channel_categories
        ), "NUM_TOTAL_CHANNELS mismatches with SEMANTIC_CHANNEL_CATEGORIES"
        semantic_channel_begin = self.num_quality_channels + 2

        need_other_categories = "others" in self.semantic_channel_categories

        construct_all_channels(
            semantic_local_points,
            self.topdown_map,
            self.current_pose,
            self.obstacle_channel,
            self.exploration_channel,
            self.quality_channel,
            semantic_channel_begin,
            self.map_cell_size,
            self.map_size_in_cells,
            self.near_threshold,
            self.far_threshold,
            self.h_min_threshold,
            self.best_scanning_distance,
            self.orientation_division_interval,
            self.semantic_channel_labels,
            self._category_semantic_sensor.category_mapping,
            need_other_categories,
        )

    @profiling_wrapper.RangeContext(
        "SemanticTopDownSensor._construct_semantic_channels"
    )
    def _construct_semantic_channels(self, semantic_map_points: torch.Tensor):
        r"""Construct semantic label information from the projected 2d points by checking
        whether there are some points belong to a specific object category that fall into
        a given cell

        :param semantic_map_points: projected 2d points in map coordinate, with semantic label in
            channel 2
        """
        num_semantic_channels = self.num_total_channels - self.num_quality_channels - 3
        assert num_semantic_channels == len(
            self.semantic_channel_categories
        ), "NUM_TOTAL_CHANNELS mismatches with SEMANTIC_CHANNEL_CATEGORIES"
        semantic_channel_begin = self.num_quality_channels + 2

        need_other_categories = "others" in self.semantic_channel_categories
        if need_other_categories:
            remained_points_mask = torch.ones(
                semantic_map_points.shape[0],
                dtype=torch.bool,
                device=semantic_map_points.device,
            )

        for channel_index, semantic_label in enumerate(self.semantic_channel_labels):
            # Some categories may not appear in this scene, which will be given
            # label -1 in _find_semantic_channel_labels()
            if semantic_label != -1:
                point_mask = semantic_map_points[..., 2] == semantic_label
                single_category_points = semantic_map_points[point_mask]

                self.topdown_map[..., semantic_channel_begin + channel_index] = (
                    self._count_points_in_each_cell(single_category_points) > 0
                )

                if need_other_categories:
                    remained_points_mask[point_mask] = False  # type: ignore

        # We leave the last channel for all "others" categories if it's specified in category names
        if need_other_categories:
            # Remove points with label 0 from remained points
            point_mask = semantic_map_points[..., 2] == 0
            remained_points_mask[point_mask] = False  # type: ignore

            remained_category_points = semantic_map_points[remained_points_mask]  # type: ignore
            self.topdown_map[
                ..., semantic_channel_begin + num_semantic_channels - 1
            ] = (self._count_points_in_each_cell(remained_category_points) > 0)

    def _project_to_topdown(self, semantic_point_cloud: torch.Tensor):
        r"""Project global point cloud and its corresponding semantic labels into a multi-layer
        2D map, with each object category occupy one channel

        :param semantic_point_cloud: point cloud with semantic label in global coordinate
        """
        # Remove y axis values from the point cloud
        projected_semantic_points = torch.hstack(
            (
                semantic_point_cloud[:, 2:3],
                semantic_point_cloud[:, 0:1],
                semantic_point_cloud[:, 3:],
            )
        )
        semantic_map_points = self._transform_global_to_map(projected_semantic_points)

        # Construct exploration channel before filtering out ground plane
        self._construct_exploration_channel(semantic_map_points)

        semantic_map_points, updated_point_heights = self._filter_too_high_or_too_low(
            semantic_map_points, semantic_point_cloud[:, 1], self.h_max_threshold
        )
        if len(semantic_map_points) < self.min_point_num_threshold:
            return self.topdown_map

        if self.use_torch_extensions:
            self._construct_quality_and_semantic_channels(semantic_map_points)
        else:
            self._construct_semantic_channels(semantic_map_points)
            self._construct_quality_channels(semantic_map_points)

        # When considering obstacle, we only care about points that lower than
        # current camera height, so we filter points once more
        semantic_map_points, _ = self._filter_too_high_or_too_low(
            semantic_map_points,
            updated_point_heights,
            cast(float, self.current_pose[1, 3].item()),
        )
        self._construct_obstacle_channel(semantic_map_points)

    def _compute_scanning_distance_score(
        self, relative_object_vectors: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute a score that represents how far each object point is scanned by the agent.
        Assuming the best scanning distance is 1.5m, scanning distance from 0.0 ~ 4.0m will
        have scores:
        ==========  ===============  =======
        Distance    Diff with best   Score
        ==========  ===============  =======
        0.0         1.5              0.4
        0.5         1.0              0.6
        1.0         0.5              0.8
        1.5         0.0              1.0
        2.0         0.5              0.8
        2.5         1.0              0.6
        3.0         1.5              0.4
        3.5         2.0              0.2
        4.0         2.5              0.0
        ==========  ===============  =======

        :param relative_object_vectors: nx2 tensor with each row represents a vector that
            points from object surface point to agent position
        :return: Tensor of (n,) shape with a scanning distance score for each point
        """
        scanning_distance = (
            torch.linalg.vector_norm(relative_object_vectors, dim=1)
            * self.map_cell_size
        )
        max_difference = max(
            self.far_threshold - self.best_scanning_distance,
            self.best_scanning_distance,
        )
        bin_length = max_difference / 5
        result = (
            max_difference - (scanning_distance - self.best_scanning_distance).abs()
        )
        result = torch.ceil(result / bin_length)
        return result / 5

    def _compute_scanning_orientation_division(
        self, relative_object_vectors: torch.Tensor
    ) -> torch.Tensor:
        r"""Divide each scanned surface point into *num_quality_channels* situations according to
        the angle it has been observed

        :param relative_object_vectors: nx2 tensor with each row represents a vector that
            points from object surface point to agent position
        :return: (n,) tensor within [0, num_quality_channels - 1] that represents which situation
            each point belongs
        """
        result = torch.atan2(
            -relative_object_vectors[:, 0], relative_object_vectors[:, 1]
        )
        result = (result + 2 * np.pi + self.orientation_division_interval / 2) % (
            2 * np.pi
        )  # Make [-theta, theta] into the first division
        return (result / self.orientation_division_interval).long()

    @profiling_wrapper.RangeContext("SemanticTopDownSensor._construct_quality_channels")
    def _construct_quality_channels(self, semantic_map_points: torch.Tensor):
        r"""Construct reconstruction quality related channels in topdown map, this part consists
        of *#num_quality_channels* channels that encode both scanning distance and orientation.
        It's useful to let the agent scan the same object from different views and appropriate
        distance

        :param semantic_map_points: projected 2d points in map coordinate, with semantic label in
            channel 2
        """
        if self.num_quality_channels == 0:
            return
        agent_position = torch.tensor(
            [self.current_pose[2, 3], self.current_pose[0, 3]],
            dtype=semantic_map_points.dtype,
            device=semantic_map_points.device,
        )
        agent_position = self._transform_global_to_map(
            agent_position.unsqueeze(0)
        ).squeeze()
        # TODO: consider whether to remove wall, floor, etc from object points
        # in mp3d dataset
        object_points = semantic_map_points[semantic_map_points[..., 2] > 0][..., :2]
        # vectors from object points to current agent position
        relative_object_vectors = (agent_position - object_points).float()
        scanning_distance_score = self._compute_scanning_distance_score(
            relative_object_vectors
        )
        scanning_orientations = self._compute_scanning_orientation_division(
            relative_object_vectors
        )

        for orientation in range(self.num_quality_channels):
            single_orientation_distance = scanning_distance_score[
                scanning_orientations == orientation
            ]
            point_positions = object_points[scanning_orientations == orientation]
            self.topdown_map[
                point_positions[:, 0],
                point_positions[:, 1],
                self.quality_channel + orientation,
            ] = single_orientation_distance

    def _compute_current_pose(self):
        world_pose = torch.from_numpy(
            construct_transformation_matrix(self._sim.get_agent_state())
        )
        if self.initial_pose_inv is None:
            self.current_pose = torch.eye(4).pin_memory()
            self.current_pose[1, 3] = self.initial_camera_height
            self.initial_pose_inv = self.current_pose @ torch.linalg.inv(world_pose)
        else:
            self.current_pose = self.initial_pose_inv @ world_pose

    @profiling_wrapper.RangeContext("SemanticTopDownSensor.get_observation")
    def get_observation(
        self, *args: Any, observations: Dict[str, Any], episode: Episode, **kwargs: Any
    ):
        try:
            self.topdown_map.zero_()  # type: ignore
        except AttributeError:
            self.topdown_map = torch.from_numpy(self.topdown_map).to(
                device=observations["depth"].device
            )

        self._check_whether_to_reset(episode)
        self._compute_current_pose()

        if self._check_ever_arrived():
            return self.topdown_map

        profiling_wrapper.range_push("semantic_channel_labels & current_pose")
        if self.use_torch_extensions:
            if not isinstance(self.semantic_channel_labels, torch.Tensor):
                self.semantic_channel_labels = torch.tensor(
                    self.semantic_channel_labels, dtype=torch.int32, device="cpu"
                ).pin_memory()
                self.semantic_channel_labels = self.semantic_channel_labels.to(
                    device=self.topdown_map.device, non_blocking=True
                )

            self.current_pose = self.current_pose.to(
                device=self.topdown_map.device, non_blocking=True
            )
        profiling_wrapper.range_pop()  # semantic_channel_labels & current_pose

        local_point_cloud = cast(
            torch.Tensor,
            self._point_cloud_sensor.get_observation(observations=observations),
        )
        profiling_wrapper.range_push("semantic_local_point_cloud")
        if self.use_torch_extensions:
            semantic_observation = observations["semantic"]
            self._category_semantic_sensor.update_instance_to_category_mapping(
                episode, semantic_observation.device
            )
        else:
            semantic_observation = self._category_semantic_sensor.get_observation(
                observations=observations, episode=episode
            )
        semantic_local_point_cloud = torch.hstack(
            (
                local_point_cloud,
                semantic_observation.reshape(
                    self._input_height * self._input_width, -1
                ),
            )
        )
        profiling_wrapper.range_pop()  # semantic_local_point_cloud

        if self.use_torch_extensions:
            self._construct_all_channels(semantic_local_point_cloud)
        else:
            semantic_local_point_cloud = self._filter_too_far_or_too_close(
                semantic_local_point_cloud
            )

            if len(semantic_local_point_cloud) < self.min_point_num_threshold:
                self.topdown_map

            semantic_global_point_cloud = self._transform_local_to_global(
                semantic_local_point_cloud
            )

            self._project_to_topdown(semantic_global_point_cloud)

        return self.topdown_map

@registry.register_sensor(name="PointCloudReconstructorSensor")
class PointCloudReconstructorSensor(Sensor):
    uuid: str = "point_cloud_reconstructor"

    def __init__(self, sim: Simulator, config: Config, **kwargs):
        super().__init__(sim=sim, config=config)
        self.sim = sim
        self.h_max = config.H_MAX
        self.f_max = config.F_MAX
        self.voxel_size = config.VOXEL_SIZE
        self._point_cloud_sensor = PointCloudCudaSensor(sim, config.POINT_CLOUD_SENSOR)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1000, 8),
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations: Dict[str, Any], episode: Episode, **kwargs: Any
    ) -> torch.Tensor:
        world_pose = construct_transformation_matrix(self.sim.get_agent_state())
        pose = torch.from_numpy(world_pose).cuda()

        point_cloud = cast(
            torch.Tensor,
            self._point_cloud_sensor.get_observation(observations=observations),
        )
        point_cloud = torch.cat(
            (
                point_cloud,
                observations["rgb"].reshape(-1, 3),
            ),
            dim=1,
        )

        # remove points that deptn == 0
        point_cloud = point_cloud[
            (point_cloud[:, 2] != 0.0) & (point_cloud[:, 2] > -self.f_max)
        ]

        point_cloud = point_cloud[point_cloud[:, 1] < self.h_max]
        point_cloud[:, 1] += 1.25

        homogeneous_xyz = torch.hstack(
            (point_cloud[:, :3], torch.ones_like(point_cloud[:, :1]))
        )
        point_cloud[:, :3] = (homogeneous_xyz @ pose.T)[:, :3]

        try:
            self.whole_point_cloud = torch.cat(
                (self.whole_point_cloud, point_cloud), dim=0
            )
        except AttributeError:
            self.whole_point_cloud = point_cloud

        self.downsample()

        return self.whole_point_cloud

    def downsample(self):
        o3d_positions = o3d.core.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(self.whole_point_cloud[:, :3])
        )
        o3d_colors = o3d.core.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(self.whole_point_cloud[:, 3:6])
        )

        o3d_point_cloud = o3d.t.geometry.PointCloud(
            {
                "positions": o3d_positions,
                "colors": o3d_colors,
            }
        )

        downsampled_point_cloud = o3d_point_cloud.voxel_down_sample(
            voxel_size=self.voxel_size
        )
        torch_positions = torch.utils.dlpack.from_dlpack(
            downsampled_point_cloud.point["positions"].to_dlpack()
        )
        torch_colors = torch.utils.dlpack.from_dlpack(
            downsampled_point_cloud.point["colors"].to_dlpack()
        )
        self.whole_point_cloud = torch.cat(
            (torch_positions, torch_colors), dim=1
        )

        torch.cuda.empty_cache()
