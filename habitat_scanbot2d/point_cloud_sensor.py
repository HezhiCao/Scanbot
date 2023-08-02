from typing import Any, Dict

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from gym import Space, spaces

from habitat.core.simulator import Sensor, Simulator, SensorTypes, DepthSensor
from habitat.config.default import Config
from habitat.core.registry import registry


@registry.register_sensor(name="PointCloudSensor")
class PointCloudSensor(Sensor):
    r"""Project agents current depth and image into a 3d point cloud

    Args:
        sim: reference to the simulator for calculating task observations.
        config:
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

    def get_observation(
        self, *args: Any, observations, **kwargs: Any
    ) -> o3d.geometry.PointCloud:
        fx = 1 / np.tan(self._input_hfov / 2.0)
        fy = self._input_width / self._input_height * fx
        intrinsic_matrix = np.array(
            [
                [fx, 0.0, 0.0, 0.0],
                [0.0, fy, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # [-1, 1] for x and [1, -1] for y as image coordinate is y-down while world is y-up
        u_grid, v_grid = np.meshgrid(
            np.linspace(-1, 1, self._input_width),
            np.linspace(1, -1, self._input_height),
        )
        depth = observations["depth"].reshape(1, self._input_height, self._input_width)
        u_grid = u_grid.reshape(1, self._input_height, self._input_width)
        v_grid = v_grid.reshape(1, self._input_height, self._input_width)

        # K(x, y, z, 1)^T = (uz, vz, z, 1)^T
        # uvz represents the right hand side matrix
        # negate depth as the camera looks along -Z
        uvz = np.vstack((u_grid * depth, v_grid * depth, -depth, np.ones(depth.shape)))
        uvz = uvz.reshape(4, -1)
        input_xyz = np.linalg.inv(intrinsic_matrix) @ uvz
        # filtered_roof_points = input_xyz[1, :] > 1.0
        # input_xyz[:3, filtered_roof_points] = 0.0
        # colorized_semantic = observations["semantic"].flatten().astype(np.int32)
        # colorized_semantic = d3_40_colors_rgb[colorized_semantic % 40].astype(np.float32) / 255.0
        input_xyz = np.transpose(input_xyz[0:3, :])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_xyz)
        pcd.colors = o3d.utility.Vector3dVector(
            observations["rgb"].reshape(-1, 3).astype(np.float32) / 255.0
        )
        orig_aabb = pcd.get_axis_aligned_bounding_box()
        epsilon = 1e-4
        new_max_bound = np.array(orig_aabb.max_bound)
        new_max_bound[2] -= epsilon
        orig_aabb.max_bound = new_max_bound
        pcd = pcd.crop(orig_aabb)
        print(pcd.get_axis_aligned_bounding_box())
        return pcd
