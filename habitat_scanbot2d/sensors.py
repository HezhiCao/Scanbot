#  ____                  _           _
# / ___|  ___ __ _ _ __ | |__   ___ | |_
# \___ \ / __/ _` | '_ \| '_ \ / _ \| __|
#  ___) | (_| (_| | | | | |_) | (_) | |_
# |____/ \___\__,_|_| |_|_.__/ \___/ \__|

"""
    habitat_scanbot2d.sensors
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Definition of scanning task related Sensors

    :copyright: (c) 2021 by XiXia.
    :license: MIT License, see LICENSE for more details.
"""

from typing import Any, Optional, Dict, Tuple, cast, TypeVar
import numpy as np
from gym import spaces, Space
import torch

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    DepthSensor,
    SemanticSensor,
    Simulator,
)
from habitat.utils import profiling_wrapper
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_scanbot2d.utils.semantic_map import (
    compute_map_size_in_cells,
    construct_transformation_matrix,
    MP3D_CATEGORY_NAMES,
)


@registry.register_sensor(name="PointCloudSensor")
class PointCloudSensor(Sensor):
    r"""Project agents current depth and image into a 3d point cloud

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointCloud sensor.
    """
    uuid: str = "point_cloud"

    def __init__(self, sim: Simulator, config: Config, **kwargs):
        self._sim = sim
        self.config = config
        self._find_depth_sensor(self._sim.sensor_suite.sensors)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._input_height * self._input_width, 3),
            dtype=np.float32,
        )

    def _find_depth_sensor(self, sensors: Dict[str, Sensor]):
        r"""Find is there a depth sensor in agent's sensor suite, and set input height, width, and
        fov of it

        :param sensors: Sensors from agent's sensor suite
        """
        depth_sensor_uuids = []
        for uuid, sensor in sensors.items():
            if isinstance(sensor, DepthSensor):
                depth_sensor_uuids.append(uuid)

        assert len(depth_sensor_uuids) == 1, (
            f"PointCloudSensor requires one Depth sensor, "
            f"{len(depth_sensor_uuids)} detected"
        )

        self._input_height = self._sim.sensor_suite.observation_spaces[
            depth_sensor_uuids[0]
        ].shape[0]
        self._input_width = self._sim.sensor_suite.observation_spaces[
            depth_sensor_uuids[0]
        ].shape[1]
        self._input_hfov = self._sim.sensor_suite.get(depth_sensor_uuids[0]).config.HFOV
        self._input_hfov = self._input_hfov * np.pi / 180.0

    def _compute_point_cloud(self, depth: np.ndarray) -> np.ndarray:
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

        # [-1, 1] for x and [1, -1] for y as image coordinate is y-down while world is y-up
        u_grid, v_grid = np.meshgrid(
            np.linspace(-1, 1, self._input_width),
            np.linspace(1, -1, self._input_height),
        )
        flatten_depth = depth.reshape(-1, 1)
        # K(x, y, z, 1)^T = (uz, vz, z, 1)^T
        # negate depth as the camera looks along -Z
        return np.hstack(
            (
                flatten_depth * u_grid.reshape(-1, 1) / fx,
                flatten_depth * v_grid.reshape(-1, 1) / fy,
                -flatten_depth,
            )
        )

    def get_observation(
        self, *args: Any, observations: Dict[str, Any], **kwargs: Any
    ) -> np.ndarray:
        return self._compute_point_cloud(observations["depth"])


@registry.register_sensor(name="SemanticTopDownSensor")
class SemanticTopDownSensor(Sensor):
    r"""Project agents current depth and semantic image into a topdown manner, this sensor will
        use PointCloudSensor and CategorySemanticSensor internally

    Args:
        sim: reference to the simulator for calculating task observations
        config: config for the SemanticTopDown sensor
    """
    uuid: str = "semantic_topdown"
    obstacle_channel = 0
    exploration_channel = 1
    quality_channel = 2

    ArrayType = TypeVar("ArrayType", np.ndarray, torch.Tensor)

    def __init__(self, sim: Simulator, config: Config, **kwargs):
        assert isinstance(
            sim, HabitatSim
        ), "This Sensor only works in HabitatSim Simulator"
        self.config = config
        self._sim = sim
        self.near_threshold = config.NEAR_THRESHOLD
        self.far_threshold = config.FAR_THRESHOLD
        self.h_min_threshold = config.H_MIN_THRESHOLD
        self.h_max_threshold = config.H_MAX_THRESHOLD
        self.initial_camera_height: float = config.INITIAL_CAMERA_HEIGHT
        self.map_size_in_meters = config.MAP_SIZE_IN_METERS
        self.map_cell_size = config.MAP_CELL_SIZE
        self.num_quality_channels = config.NUM_QUALITY_CHANNELS
        self.orientation_division_interval = (
            2 * np.pi / self.num_quality_channels
            if self.num_quality_channels > 0
            else -1.0
        )
        self.best_scanning_distance = config.BEST_SCANNING_DISTANCE
        self.num_total_channels = config.NUM_TOTAL_CHANNELS
        self.semantic_channel_categories = config.SEMANTIC_CHANNEL_CATEGORIES
        self.dataset_type = config.DATASET_TYPE
        self.min_point_num_threshold = config.MIN_POINT_NUM_THRESHOLD
        self.map_size_in_cells = compute_map_size_in_cells(
            self.map_size_in_meters, self.map_cell_size
        )
        self.initial_pose_inv: Optional[np.ndarray] = None
        self.current_pose: np.ndarray
        self.arrived_poses = set()
        self._current_episode_id: Optional[str] = None
        self._point_cloud_sensor = PointCloudSensor(sim, config.POINT_CLOUD_SENSOR)
        self._category_semantic_sensor = CategorySemanticSensor(
            sim, config.CATEGORY_SEMANTIC_SENSOR
        )
        self.topdown_map = np.zeros(
            (self.map_size_in_cells, self.map_size_in_cells, self.num_total_channels),
            dtype=np.float32,
        )
        self._find_sensor_pair(self._sim.sensor_suite.sensors)
        super().__init__(config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(
                self.map_size_in_cells,
                self.map_size_in_cells,
                self.num_total_channels,
            ),
            dtype=np.float32,
        )

    def _find_sensor_pair(self, sensors: Dict[str, Sensor]):
        r"""Find is there a pair of depth and semantic sensors in agent's sensor suite, and set
        input height, width of them

        :param sensors: Sensors from agent's sensor suite
        """
        depth_sensor_uuids = []
        semantic_sensor_uuids = []
        for uuid, sensor in sensors.items():
            if isinstance(sensor, DepthSensor):
                depth_sensor_uuids.append(uuid)
            if isinstance(sensor, SemanticSensor):
                semantic_sensor_uuids.append(uuid)

        assert len(depth_sensor_uuids) == 1, (
            f"SemanticTopDownSensor requires one Depth sensor, "
            f"{len(depth_sensor_uuids)} detected"
        )

        assert len(semantic_sensor_uuids) == 1, (
            f"SemanticTopDownSensor requires one Semantic sensor, "
            f"{len(semantic_sensor_uuids)} detected"
        )

        assert (
            self._sim.sensor_suite.observation_spaces[depth_sensor_uuids[0]].shape[:2]
            == self._sim.sensor_suite.observation_spaces[
                semantic_sensor_uuids[0]
            ].shape[:2]
        ), "SemanticTopDownSensor requires matched Depth and Semantic sensors"

        self._input_height = self._sim.sensor_suite.observation_spaces[
            depth_sensor_uuids[0]
        ].shape[0]
        self._input_width = self._sim.sensor_suite.observation_spaces[
            depth_sensor_uuids[0]
        ].shape[1]

    def _find_semantic_channel_labels(self):
        r"""Find a semantic channel labels list from descriptive category names

        :param dataset_type: what type of scene datasets will be used
        """
        # Note that correspondence between category names and category labels is not
        # fixed from scene to scene in gibson dataset
        if self.dataset_type == "gibson":
            self.semantic_channel_labels = [-1] * (
                len(self.semantic_channel_categories) - 1
                if "others" in self.semantic_channel_categories
                else len(self.semantic_channel_categories)
            )
            objects = self._sim.semantic_annotations().objects
            for obj in objects:
                if obj is not None:
                    try:
                        self.semantic_channel_labels[
                            self.semantic_channel_categories.index(obj.category.name())
                        ] = obj.category.index()
                    except ValueError:
                        pass

        # However, in Matterport3D it's fixed, so we can use a predefined mapping
        # And only compute it once
        elif self.dataset_type == "mp3d":
            if not hasattr(self, "semantic_channel_labels"):
                self.semantic_channel_labels = [
                    MP3D_CATEGORY_NAMES[name]
                    for name in self.semantic_channel_categories
                    if name != "others"
                ]
        else:
            raise NotImplementedError

    def _transform_local_to_global(self, point_cloud: np.ndarray) -> np.ndarray:
        r"""Transform point cloud in camera coordinate to global coordinate

        :param point_cloud: point cloud in camera coordinate, **note that** input point cloud may
            contain not only xyz, so we will only change the first three channels
        :return: transformed point cloud
        """
        local_xyz = point_cloud[:, :3]
        homogeneous_xyz = np.hstack((local_xyz, np.ones_like(local_xyz[:, :1])))
        point_cloud[:, :3] = (homogeneous_xyz @ self.current_pose.T)[:, :3]
        return point_cloud

    def _filter_too_far_or_too_close(self, point_cloud: np.ndarray) -> np.ndarray:
        r"""Remove points that are too far or too close to the camera

        :param point_cloud: point cloud in camera coordinate
        :return: filtered point cloud
        """
        medium_range_indexes = (np.abs(point_cloud[:, 2]) < self.far_threshold) & (
            np.abs(point_cloud[:, 2]) > self.near_threshold
        )
        return point_cloud[medium_range_indexes]

    def _filter_too_high_or_too_low(
        self, map_points: ArrayType, point_heights: ArrayType, h_max_threshold: float
    ) -> Tuple[ArrayType, ArrayType]:
        r"""Remove points that are too high or too low in the scene, but not from the camera's
        perspective, since the camera position may change during episode

        :param map_points: projected 2d points in map coordinate
        :param h_max_threshold: threshold for maximum height that will be kept
        :return: filtered map points
        """
        medium_range_indexes = (point_heights < h_max_threshold) & (
            point_heights > self.h_min_threshold
        )
        return map_points[medium_range_indexes], point_heights[medium_range_indexes]  # type: ignore

    def _transform_global_to_map(self, projected_points: np.ndarray) -> np.ndarray:
        r"""Projected 2d points in global coordinate into map coordinate by applying a scaling
        and a translation transformation

        :param projected_points: projected 2d points without y values in global coordinate
        :return: HxWxC array with first two channels represent (row, col)
        """
        translation = self.map_size_in_cells // 2
        projected_points[:, :2] = (
            projected_points[:, :2] / self.map_cell_size + translation
        )
        result = projected_points.astype(np.int32)
        np.clip(
            result[:, :2], a_min=0, a_max=self.map_size_in_cells - 1, out=result[:, :2]
        )
        return result

    def _count_points_in_each_cell(self, map_points: np.ndarray) -> np.ndarray:
        r"""Count how many points belong to a individual cell

        :param map_points: projected 2d points in map coordinate
        :param topdown_map: HxW array to be filled, echo cell represents the count of
            points falling into it
        """
        flattened_indexes = map_points[:, 0] * self.map_size_in_cells + map_points[:, 1]
        point_count_map = np.bincount(
            flattened_indexes, minlength=self.map_size_in_cells ** 2
        )
        return point_count_map.reshape(
            self.map_size_in_cells, self.map_size_in_cells
        ).astype(np.float32)

    @profiling_wrapper.RangeContext("SemanticTopDownSensor._construct_obstacle_channel")
    def _construct_obstacle_channel(self, map_points: ArrayType):
        r"""Construct obstacle information from the projected 2d points by counting
        how many points belong to a individual cell

        :param map_points: projected 2d points in map coordinate
        :param topdown_map: HxWxC array to be filled, echo cell in obstacle channel represents
            the count of points falling into it
        """
        self.topdown_map[..., self.obstacle_channel] = self._count_points_in_each_cell(
            map_points  # type: ignore
        )

    @profiling_wrapper.RangeContext(
        "SemanticTopDownSensor._construct_exploration_channel"
    )
    def _construct_exploration_channel(self, map_points: ArrayType):
        r"""Construct exploration information from the projected 2d points by checking
        whether there is a point belongs to a individual cell

        :param map_points: projected 2d points in map coordinate
        :param topdown_map: HxWxC array to be filled, echo cell in exploration channel represents
            whether this cell has been explored
        """
        self.topdown_map[
            map_points[:, 0],
            map_points[:, 1],
            self.exploration_channel,
        ] = 1.0

    def _construct_semantic_channels(self, semantic_map_points: np.ndarray):
        r"""Construct semantic label information from the projected 2d points by checking
        whether there are some points belong to a specified object category that fall into
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
            remained_points_mask = np.ones(semantic_map_points.shape[0], dtype=bool)

        for channel_index, semantic_label in enumerate(self.semantic_channel_labels):
            # Some categories may not appear in this scene, which will be given
            # label -1 in _find_semantic_channel_labels()
            if semantic_label != -1:
                point_mask = semantic_map_points[..., 2] == semantic_label
                single_category_points = semantic_map_points[point_mask]

                self.topdown_map[..., semantic_channel_begin + channel_index] = (
                    self._count_points_in_each_cell(single_category_points) > 0
                ).astype(np.float32)

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
            ] = (self._count_points_in_each_cell(remained_category_points) > 0).astype(
                np.float32
            )

    def _project_to_topdown(self, semantic_point_cloud: np.ndarray):
        r"""Project global point cloud and its corresponding semantic labels into a multi-layer
        2D map, with each object category occupy one channel

        :param semantic_point_cloud: point cloud with semantic label in global coordinate
        """
        # Remove y axis values from the point cloud
        projected_semantic_points = np.hstack(
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
        self._construct_semantic_channels(semantic_map_points)

        self._construct_quality_channels(semantic_map_points)

        # When considering obstacle, we only care about points that lower than
        # current camera height, so we filter points once more
        semantic_map_points, _ = self._filter_too_high_or_too_low(
            semantic_map_points, updated_point_heights, self.current_pose[1, 3]
        )
        self._construct_obstacle_channel(semantic_map_points)

    def _compute_scanning_distance_score(
        self, relative_object_vectors: np.ndarray
    ) -> np.ndarray:
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
            np.linalg.norm(relative_object_vectors, axis=1) * self.map_cell_size
        )
        max_difference = max(
            self.far_threshold - self.best_scanning_distance,
            self.best_scanning_distance,
        )
        bin_length = max_difference / 5
        result = max_difference - np.abs(
            scanning_distance - self.best_scanning_distance
        )
        result = np.ceil(result / bin_length)
        return result / 5

    def _compute_scanning_orientation_division(
        self, relative_object_vectors: np.ndarray
    ) -> np.ndarray:
        r"""Divide each scanned surface point into *num_quality_channels* situations according to
        the angle it has been observed

        :param relative_object_vectors: nx2 array with each row represents a vector that
            points from object surface point to agent position
        :return: (n,) array within [0, num_quality_channels - 1] that represents which situation
            each point belongs
        """
        result = np.arctan2(
            -relative_object_vectors[:, 0], relative_object_vectors[:, 1]
        )
        result = (result + 2 * np.pi + self.orientation_division_interval / 2) % (
            2 * np.pi
        )  # Make [-theta, theta] into the first division
        return (result / self.orientation_division_interval).astype(np.int32)

    def _construct_quality_channels(self, semantic_map_points: np.ndarray):
        r"""Construct reconstruction quality related channels in topdown map, this part consists
        of *#num_quality_channels* channels that encode both scanning distance and orientation.
        It's useful to let the agent scan the same object from different views and appropriate
        distance

        :param semantic_map_points: projected 2d points in map coordinate, with semantic label in
            channel 2
        """
        if self.num_quality_channels == 0:
            return
        agent_position = np.array(
            [self.current_pose[2, 3], self.current_pose[0, 3]],
            dtype=semantic_map_points.dtype,
        )
        agent_position = self._transform_global_to_map(
            agent_position[np.newaxis, :]
        ).squeeze()
        object_points = semantic_map_points[semantic_map_points[..., 2] > 0][..., :2]
        # vectors from object points to current agent position
        relative_object_vectors = (agent_position - object_points).astype(np.float32)
        scanning_distance_score = self._compute_scanning_distance_score(
            relative_object_vectors
        )
        scanning_orientations = self._compute_scanning_orientation_division(
            relative_object_vectors
        )

        quality_channel_begin = 2
        for orientation in range(self.num_quality_channels):
            single_orientation_distance = scanning_distance_score[
                scanning_orientations == orientation
            ]
            point_positions = object_points[scanning_orientations == orientation]
            self.topdown_map[
                point_positions[:, 0],
                point_positions[:, 1],
                quality_channel_begin + orientation,
            ] = single_orientation_distance

    @profiling_wrapper.RangeContext("SemanticTopDownSensor._check_ever_arrived")
    def _check_ever_arrived(self) -> bool:
        r"""Check a pose has already been arrived by the agent, if not add it into
        a hash set
        """
        current_forward = self.current_pose @ np.append(self._sim.forward_vector, 0)
        pose_repr = (
            int(self.current_pose[2, 3] / self.map_cell_size),
            int(self.current_pose[0, 3] / self.map_cell_size),
            int(
                (
                    np.arctan2(-current_forward[2], current_forward[0])
                    + self.orientation_division_interval / 2
                )
                / self.orientation_division_interval
            ),
        )
        if pose_repr in self.arrived_poses:
            return True
        else:
            self.arrived_poses.add(pose_repr)
            return False

    def _check_whether_to_reset(self, episode: Episode):
        r"""Reset internal states when episode_id changed

        :param episode: current experiencing episode
        """
        if self._current_episode_id != episode.episode_id:
            self._current_episode_id = episode.episode_id
            self.initial_pose_inv = None
            self.arrived_poses = set()
            self._find_semantic_channel_labels()

    def _compute_current_pose(self):
        world_pose = construct_transformation_matrix(self._sim.get_agent_state())
        if self.initial_pose_inv is None:
            self.current_pose = np.eye(4)
            self.current_pose[1, 3] = self.initial_camera_height
            self.initial_pose_inv = self.current_pose @ np.linalg.inv(world_pose)
        else:
            self.current_pose = self.initial_pose_inv @ world_pose

    def get_observation(
        self, *args: Any, observations: Dict[str, Any], episode: Episode, **kwargs: Any
    ):
        self.topdown_map.fill(0.0)
        self._check_whether_to_reset(episode)
        self._compute_current_pose()

        if self._check_ever_arrived():
            return self.topdown_map

        local_point_cloud = self._point_cloud_sensor.get_observation(
            observations=observations
        )
        semantic_observation = self._category_semantic_sensor.get_observation(
            observations=observations, episode=episode
        )
        semantic_local_point_cloud = np.hstack(
            (
                local_point_cloud,
                semantic_observation.reshape(
                    self._input_height * self._input_width, -1
                ),
            )
        )

        semantic_local_point_cloud = self._filter_too_far_or_too_close(
            semantic_local_point_cloud
        )

        if len(semantic_local_point_cloud) < self.min_point_num_threshold:
            return self.topdown_map

        semantic_global_point_cloud = self._transform_local_to_global(
            semantic_local_point_cloud
        )

        self._project_to_topdown(semantic_global_point_cloud)
        return self.topdown_map


@registry.register_sensor
class MapPoseSensor(Sensor):
    r"""Sensor for current agent pose in the max pooled global map that built from
    SemanticTopDownSensor or SemanticMapBuilder

    Args:
        sim: reference to the simulator for querying current agent state
        config: config for the MapPose sensor. Have to contain the field
            - MAP_SIZE_IN_METERS: max size of the scene that the map can
                represent
            - MAP_CELL_SIZE: the real size that one cell represents
    """
    uuid: str = "map_pose"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        assert isinstance(
            sim, HabitatSim
        ), "This Sensor only works in HabitatSim Simulator"
        self._sim = sim
        self.map_size_in_meters = config.MAP_SIZE_IN_METERS
        self.map_cell_size = config.MAP_CELL_SIZE
        self.map_size_in_cells = compute_map_size_in_cells(
            self.map_size_in_meters, self.map_cell_size
        )
        self.initial_pose_inv: Optional[np.ndarray] = None
        self._current_episode_id: Optional[str] = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.array((0.0, 0.0, -1.0, -1.0)),
            high=np.array(
                (
                    self.map_size_in_cells - 1,
                    self.map_size_in_cells - 1,
                    1.0,
                    1.0,
                ),
            ),
            shape=(4,),
            dtype=np.float32,
        )

    def _compute_map_position(self, pose) -> np.ndarray:
        r"""Compute the 2d position in map pixel from a 3d 4x4 pose matrix, assuming the origin
        is at the map center

        :param pose: 4x4 pose matrix relative to the first frame camera coordinate
        :return: (row, col) vector
        """
        translation = self.map_size_in_cells // 2
        result = np.array(
            (
                pose[2, 3] / self.map_cell_size + translation,
                pose[0, 3] / self.map_cell_size + translation,
            ),
            dtype=np.float32,
        )
        np.clip(result, a_min=0.0, a_max=self.map_size_in_cells - 1, out=result)
        return result

    def _compute_forward_direction(self, pose) -> np.ndarray:
        r"""Compute the forward direction of the agent in 2d map from a 3d 4x4 pose matrix, assuming
        the origin is at the map center

        :param pose: 4x4 pose matrix relative to the first frame camera coordinate
        :return: 2d forward direction vector
        """
        current_forward = pose @ np.append(self._sim.forward_vector, 0)
        current_forward = np.array(
            (current_forward[2], current_forward[0]), dtype=np.float32
        )
        return current_forward / np.linalg.norm(current_forward)

    def convert_map_coordinate_to_world_coordinate(self, map_point) -> np.ndarray:
        r"""Convert a 2d point in map coordinate into a 3d point in
        world coordinate

        :param map_point: point in map coordinate to be converted
        :return: converted 3d point in world coordinate
        """
        translation = self.map_size_in_cells // 2
        map_point = (map_point - translation) * self.map_cell_size
        return (self.initial_pose @ np.array([map_point[1], 0.0, map_point[0], 1.0]))[
            :3
        ]

    def get_observation(self, *args: Any, episode: Episode, **kwargs: Any) -> Any:
        if self._current_episode_id != episode.episode_id:
            self._current_episode_id = episode.episode_id
            self.initial_pose_inv = None

        world_pose = construct_transformation_matrix(self._sim.get_agent_state())
        if self.initial_pose_inv is None:
            self.initial_pose = world_pose
            self.initial_pose_inv = np.linalg.inv(world_pose)
            pose = np.eye(4)
        else:
            pose = self.initial_pose_inv @ world_pose

        map_position = self._compute_map_position(pose)
        forward_direction = self._compute_forward_direction(pose)
        return np.concatenate((map_position, forward_direction))


@registry.register_sensor(name="CategorySemanticSensor")
class CategorySemanticSensor(SemanticSensor):
    r"""Provide semantic labels corresponding to object categories instead of object instances

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the CategorySemantic sensor.
    """
    uuid = "category_semantic"

    def __init__(self, sim: Simulator, config: Config, **kwargs):
        self._sim = cast(HabitatSim, sim)
        self.config = config
        self._current_scene_id: Optional[str] = None
        self.category_mapping: np.ndarray
        self._find_instance_semantic_sensor(self._sim.sensor_suite.sensors)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            shape=(self._input_height, self._input_width),
            dtype=np.int32,
        )

    def _find_instance_semantic_sensor(self, sensors: Dict[str, Sensor]):
        r"""Find is there a default semantic sensor (instance label) in agent's sensor suite

        :param sensors: Sensors from agent's sensor suite
        """
        semantic_sensor_uuids = []
        for uuid, sensor in sensors.items():
            if isinstance(sensor, SemanticSensor):
                semantic_sensor_uuids.append(uuid)

        assert len(semantic_sensor_uuids) == 1, (
            f"CategorySemanticSensor requires one Instance Semantic sensor, "
            f"{len(semantic_sensor_uuids)} detected"
        )

        self._input_height = self._sim.sensor_suite.observation_spaces[
            semantic_sensor_uuids[0]
        ].shape[0]
        self._input_width = self._sim.sensor_suite.observation_spaces[
            semantic_sensor_uuids[0]
        ].shape[1]

    def _update_instance_to_category_mapping(self, episode: Episode):
        r"""Build a mapping from instance label to category label in current scene"""
        self._current_scene_id = episode.scene_id

        scene = self._sim.semantic_annotations()
        instance_id_to_category_id = {
            int(obj.id.split("_")[-1]): obj.category.index()
            for obj in scene.objects
            if obj is not None
        }
        first_instance_label = min(instance_id_to_category_id.keys())
        # Make 0 label remains 0, if instance label start from 1
        self.category_mapping = np.array(
            [0] * first_instance_label
            + [
                instance_id_to_category_id[
                    i + first_instance_label
                ]  # instance id may start from 1
                for i in range(len(instance_id_to_category_id))
            ],
            dtype=np.int32,
        )

    def get_observation(
        self, *args: Any, observations: Dict[str, Any], episode: Episode, **kwargs: Any
    ) -> np.ndarray:
        if self._current_scene_id != episode.scene_id:
            self._update_instance_to_category_mapping(episode)
        return np.take(self.category_mapping, observations["semantic"])
