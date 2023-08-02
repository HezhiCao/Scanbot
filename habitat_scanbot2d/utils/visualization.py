"""
    habitat_scanbot2d.visualization
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Collection of some visualization utilities

    :copyright: (c) 2021 by XiXia.
    :license: MIT License, see LICENSE for more details.
"""

from typing import Callable, Dict, Optional, Union, List, cast
import numpy as np
import cv2
import torch
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Ellipse
from matplotlib.transforms import Bbox
import cmasher as cmr

import habitat
from habitat_sim.utils.viz_utils import observation_to_image
from habitat_sim.utils.common import d3_40_colors_rgb
from habitat.utils.visualizations.utils import tile_images
from habitat.utils.visualizations import maps
from habitat_scanbot2d.semantic_map_builder import SemanticMapBuilder
from habitat_scanbot2d.sensors import SemanticTopDownSensor, MapPoseSensor
from habitat_scanbot2d.cuda_sensors import SemanticTopDownCudaSensor

# pylint: disable=no-member


try:
    # o3d may initialize cuda context in multiple devices
    # to save memory we only import it and define related
    # functions at local machine
    import open3d as o3d

    def construct_o3d_point_cloud(
        xyz_points: np.ndarray, rgb: np.ndarray
    ) -> o3d.geometry.PointCloud:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz_points)
        point_cloud.colors = o3d.utility.Vector3dVector(
            rgb.reshape(-1, 3).astype(np.float32) / 255.0
        )
        return point_cloud

    def visualize_point_cloud(xyz_points: np.ndarray, rgb: np.ndarray):
        o3d.visualization.draw_geometries([construct_o3d_point_cloud(xyz_points, rgb)])


except:
    pass


def visualize_semantic_point_cloud(point_cloud_with_semantic: np.ndarray):
    semantic_labels = point_cloud_with_semantic[:, -1].astype(np.int32)
    visualize_point_cloud(
        point_cloud_with_semantic[:, :3], d3_40_colors_rgb[semantic_labels % 40]
    )


def visualize_rgb(rgb):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
    ax.imshow(rgb)
    ax.set_title("Rgb image")
    plt.show()


def visualize_semantic(semantic):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
    ax.imshow(observation_to_image(semantic, "semantic"))
    ax.set_title("Semantic image")
    plt.show()


def visualize_single_channel_image(image):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
    ax.imshow(image)
    ax.set_title("Single channel image")
    plt.show()


def visualize_depth_semantic(depth, semantic):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 2, 1)
    ax.axis("off")
    ax.set_title("Depth image")
    ax.imshow(observation_to_image(depth.squeeze(), "depth"))
    ax = plt.subplot(1, 2, 2)
    ax.axis("off")
    ax.set_title("Semantic image")
    ax.imshow(observation_to_image(semantic, "semantic"))
    plt.show()


def construct_fancy_arrow(agent_pose: np.ndarray):
    agent_size = 16
    return FancyArrow(
        agent_pose[1],
        agent_pose[0],
        agent_pose[3] * agent_size,
        agent_pose[2] * agent_size,
        head_width=agent_size,
        head_length=agent_size * 1.25,
        length_includes_head=True,
        alpha=0.8,
        color="#BAE67E",
    )


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    :param observation: observation returned from an environment step().
    :param info: info returned from an environment step().

    :return: generated image of a single frame.
    """
    render_obs_images: List[np.ndarray] = []
    for sensor_name in observation:
        if "rgb" in sensor_name:
            rgb = observation[sensor_name]
            if not isinstance(rgb, np.ndarray):
                rgb = rgb.cpu().numpy()

            render_obs_images.append(rgb)
        elif "depth" in sensor_name:
            depth = observation[sensor_name].squeeze() * 255.0
            if not isinstance(depth, np.ndarray):
                depth = depth.cpu().numpy()

            depth = depth.astype(np.uint8)
            depth = np.stack([depth for _ in range(3)], axis=2)
            render_obs_images.append(depth)

    assert len(render_obs_images) > 0, "Expected at least one visual sensor enabled."

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)

    semantic_map = (
        observation["semantic_topdown"]
        if "semantic_topdown" in observation
        else observation["semantic_topdown_cuda"].cpu().numpy()
    )

    top_down_map = (
        np.sum(
            semantic_map[
                ...,
                SemanticTopDownSensor.quality_channel : SemanticTopDownSensor.quality_channel
                + 8,
            ],
            axis=-1,
        )
        / 4
    )

    plasma_cmap = plt.cm.get_cmap("plasma")
    top_down_map = (plasma_cmap(top_down_map)[..., :3] * 255.0).astype(np.uint8)  # type: ignore

    agent_pose = observation["map_pose"].cpu().numpy()

    agent_center_coord = ((agent_pose[:2] + 1.0) * top_down_map.shape[0] / 2.0).astype(
        np.int32
    )
    agent_rotation = np.arctan2(*agent_pose[2:])
    top_down_map = maps.draw_agent(top_down_map, agent_center_coord, agent_rotation)

    render_frame = cv2.resize(
        render_frame,
        (
            top_down_map.shape[0],
            int(top_down_map.shape[0] / render_frame.shape[0] * render_frame.shape[1]),
        ),
        interpolation=cv2.INTER_LINEAR,
    )
    render_frame = np.concatenate((render_frame, top_down_map), axis=1)
    return render_frame


class SemanticMapViewer:
    def __init__(
        self,
        observations: Dict[str, Union[np.ndarray, torch.Tensor]],
        num_channels: int,
        category_name_map: Dict[int, str],
        normalize_obstacle_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        env: Optional[habitat.Env] = None,
        map_builder: Optional[SemanticMapBuilder] = None,
    ) -> None:
        rcParams["keymap.xscale"] = ["L"]  # remove k from default keymap
        rcParams["keymap.zoom"] = ["z"]  # remove o from default keymap
        rcParams["keymap.save"] = ["ctrl-s"]  # remove s from default keymap
        rcParams["keymap.fullscreen"] = ["ctrl-f"]  # remove s from default keymap
        self._semantic_map_observation_uuid = "global_semantic_map"
        self._front_view_observation_uuids = ("rgb", "depth", "semantic")
        self._semantic_map = (
            observations[SemanticTopDownCudaSensor.uuid]
            if SemanticTopDownCudaSensor.uuid in observations
            else observations[SemanticTopDownSensor.uuid]
        )
        self._front_view_observation = observations[
            self._front_view_observation_uuids[0]
        ]
        self._num_channels = num_channels + 1  # add extra integrated quality channel
        self._normalize_obstacle_fn = normalize_obstacle_fn
        self.category_name_map = category_name_map
        self._current_channel = 0
        self._current_front_view = 0
        self._path: Optional[List[np.ndarray]] = None
        self._goal_position: Optional[np.ndarray] = None
        self.goal_mean: Optional[np.ndarray] = None
        self.goal_stddev: Optional[np.ndarray] = None

        if "local_semantic_map" in observations:
            self.use_local_representation = True
            self.scale_factor = int(
                observations["global_semantic_map"].shape[0]
                / observations["local_semantic_map"].shape[0]
            )
        else:
            self.use_local_representation = False

        self._preprocess_observations()
        self._construct_figure_and_axes()

        if env is not None and map_builder is not None:
            self.map_builder = map_builder
            self.env = env
            self.agent_pose = cast(np.ndarray, observations[MapPoseSensor.uuid])
            self.topdown_axes.add_patch(construct_fancy_arrow(self.agent_pose))

        self.fig.canvas.mpl_connect("key_press_event", self)

    def _preprocess_observations(self):
        if isinstance(self._semantic_map, torch.Tensor):
            self._semantic_map = self._semantic_map.cpu().numpy()
            if self._normalize_obstacle_fn is not None:
                self._semantic_map[
                    ..., SemanticTopDownSensor.obstacle_channel
                ] = self._normalize_obstacle_fn(
                    self._semantic_map[..., SemanticTopDownSensor.obstacle_channel]
                )

        if isinstance(self._front_view_observation, torch.Tensor):
            self._front_view_observation = self._front_view_observation.cpu().numpy()

        if (
            self.use_local_representation
            and self._semantic_map_observation_uuid == "local_semantic_map"
        ):
            try:
                self._semantic_map = self._semantic_map.repeat(
                    self.scale_factor, axis=0
                ).repeat(self.scale_factor, axis=1)
            except AttributeError:
                pass

    def _construct_figure_and_axes(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.topdown_axes, self.front_view_axes = self.fig.subplots(1, 2)
        self.topdown_image = self.topdown_axes.imshow(
            self._semantic_map[..., SemanticTopDownSensor.obstacle_channel],
            vmin=0.0,
            vmax=1.0,
        )
        self.topdown_axes.set_xlim(left=0, right=self._semantic_map.shape[1])
        self.topdown_axes.set_ylim(top=0, bottom=self._semantic_map.shape[0])
        self.topdown_axes.set_title("Obstacle channel")
        self.front_view_image = self.front_view_axes.imshow(
            self._front_view_observation,
            cmap=cmr.ember,
        )
        self.front_view_axes.set_title("Current RGB")

    @property
    def path(self) -> Optional[List[np.ndarray]]:
        return self._path

    @path.setter
    def path(self, path: List[np.ndarray]):
        self._path = path

    @property
    def goal_position(self) -> Optional[np.ndarray]:
        return self._goal_position

    @goal_position.setter
    def goal_position(self, goal_position: Optional[np.ndarray]):
        self._goal_position = goal_position

    def update_observations(self, obs):
        try:
            self.map_builder.update(obs)  # type: ignore
        except AttributeError:
            pass
        self._semantic_map = obs[self._semantic_map_observation_uuid]
        self._front_view_observation = obs[
            self._front_view_observation_uuids[self._current_front_view]
        ]
        self._preprocess_observations()
        self.agent_pose = obs[MapPoseSensor.uuid]

    def __call__(self, event):
        if event.key == "j":
            self._current_channel = (self._current_channel + 1) % self._num_channels
        elif event.key == "k":
            self._current_channel = (self._current_channel - 1) % self._num_channels
        elif event.key == "w":
            obs = self.env.step({"action": "MOVE_FORWARD", "action_args": None})
            self.update_observations(obs)
        elif event.key == "a":
            obs = self.env.step({"action": "TURN_LEFT", "action_args": None})
            self.update_observations(obs)
        elif event.key == "d":
            obs = self.env.step({"action": "TURN_RIGHT", "action_args": None})
            self.update_observations(obs)
        elif event.key == "f":
            obs = self.env.step({"action": "LOOK_DOWN", "action_args": None})
            self.update_observations(obs)
        elif event.key == "b":
            obs = self.env.step({"action": "LOOK_UP", "action_args": None})
            self.update_observations(obs)
        elif event.key == "c":
            if self.use_local_representation:
                self._semantic_map_observation_uuid = (
                    "local_semantic_map"
                    if self._semantic_map_observation_uuid == "global_semantic_map"
                    else "global_semantic_map"
                )
        elif event.key == "o":
            self._current_front_view = (self._current_front_view + 1) % len(
                self._front_view_observation_uuids
            )
        elif event.key == "s":
            np.save("semantic_map.npy", self._semantic_map)
            try:
                self.map_builder.save()
            except AttributeError:
                pass
        else:
            return
        self.update_figure()

    def _update_topdown_image(self):
        if self._current_channel < self._num_channels - 1:
            self.topdown_image.set_data(self._semantic_map[..., self._current_channel])
        else:
            self.topdown_image.set_data(
                np.sum(
                    self._semantic_map[
                        ...,
                        SemanticTopDownSensor.quality_channel : SemanticTopDownSensor.quality_channel
                        + 8,
                    ],
                    axis=-1,
                )
                / 4
            )

        if self._current_channel == SemanticTopDownSensor.obstacle_channel:
            self.topdown_axes.set_title("Obstacle channel")
        elif self._current_channel == SemanticTopDownSensor.exploration_channel:
            self.topdown_axes.set_title("Exploration channel")
        elif self._current_channel < 10:
            self.topdown_axes.set_title(
                f"Quality channel {self._current_channel - SemanticTopDownSensor.quality_channel}"
            )
        elif self._current_channel == self._num_channels - 2:
            self.topdown_axes.set_title("Trajectory channel")
        # add one extra channel for integrated quality
        elif self._current_channel == self._num_channels - 1:
            self.topdown_axes.set_title("Integrated quality")
        else:
            self.topdown_axes.set_title(
                "Semantic channel " + self.category_name_map[self._current_channel - 10]
            )

    def _update_agent_arrow(self):
        agent_pose = self.agent_pose.copy()
        if self._semantic_map_observation_uuid == "local_semantic_map":
            agent_pose[:2] = self._semantic_map.shape[0] // 2
        try:
            self.topdown_axes.patches.clear()
            self.topdown_axes.add_patch(construct_fancy_arrow(agent_pose))
        except AttributeError:
            pass

    def _update_front_view_image(self):
        if self._front_view_observation_uuids[self._current_front_view] == "rgb":
            self.front_view_axes.set_title("Current RGB")
            self.front_view_image.set_data(self._front_view_observation)
        elif self._front_view_observation_uuids[self._current_front_view] == "depth":
            self.front_view_axes.set_title("Current Depth")
            self.front_view_image.set_data(
                observation_to_image(
                    cast(np.ndarray, self._front_view_observation).squeeze(), "depth"
                )
            )
        else:
            self.front_view_axes.set_title("Current Semantic")
            self.front_view_image.set_data(
                observation_to_image(
                    cast(np.ndarray, self._front_view_observation), "semantic"
                )
            )

    def _update_goal_scatter_and_range(self):
        try:
            self.goal_scatter.remove()
            self.goal_mean_scatter.remove()
        except:
            pass

        if self.goal_position is not None:
            goal_position = self._convert_goal_point_to_image_point(self.goal_position)
            self.goal_scatter = self.topdown_axes.scatter(
                goal_position[1],
                goal_position[0],
                s=50,
                c="#5CCFE6",
                clip_on=True,
            )

        if self.goal_mean is not None and self.goal_stddev is not None:
            goal_mean = self._convert_goal_point_to_image_point(self.goal_mean)
            self.goal_mean_scatter = self.topdown_axes.scatter(
                goal_mean[1],
                goal_mean[0],
                s=10,
                c="#FFCC66",
                clip_on=True,
            )
            # 95% within two standard deviation
            ellipse_width = self._semantic_map.shape[1] * self.goal_stddev[1] * 2
            ellipse_height = self._semantic_map.shape[0] * self.goal_stddev[0] * 2
            if self._semantic_map_observation_uuid == "global_semantic_map":
                ellipse_width /= self.scale_factor
                ellipse_height /= self.scale_factor

            stddev_ellipse = Ellipse(
                (goal_mean[1], goal_mean[0]),
                ellipse_width,
                ellipse_height,
                alpha=0.3,
                color="#D4BFFF",
                clip_on=True,
            )
            self.topdown_axes.add_patch(stddev_ellipse)

    def _update_path_lines(self):
        if self.path is not None:
            self.topdown_axes.lines.clear()
            if len(self.path) == 1:
                self.path = [self.agent_pose[:2]] + self.path
            if self._semantic_map_observation_uuid == "local_semantic_map":
                path = np.array(self.path, dtype=np.int32)
                path = (
                    self.scale_factor * (path - self.agent_pose[:2])
                    + np.array(self._semantic_map.shape[:2]) // 2
                )
                np.clip(path, a_min=0, a_max=self._semantic_map.shape[0] - 1, out=path)
                self.topdown_axes.plot(
                    [path_point[1] for path_point in path],
                    [path_point[0] for path_point in path],
                    "r",
                )
            else:
                self.topdown_axes.plot(
                    [path_point[1] for path_point in self.path],
                    [path_point[0] for path_point in self.path],
                    "r",
                )

    def update_figure(self):
        self._update_topdown_image()
        self._update_front_view_image()
        self._update_agent_arrow()
        self._update_goal_scatter_and_range()
        self._update_path_lines()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _convert_goal_point_to_image_point(self, goal_point: np.ndarray) -> np.ndarray:
        goal_point = goal_point.copy()
        if self._semantic_map_observation_uuid == "local_semantic_map":
            goal_point = (
                self.scale_factor * (goal_point - self.agent_pose[:2].astype(np.int32))
                + np.array(self._semantic_map.shape[:2], dtype=np.int32) // 2
            )
            np.clip(
                goal_point,
                a_min=0,
                a_max=self._semantic_map.shape[0] - 1,
                out=goal_point,
            )
        return goal_point


class CompletedRateViewer(SemanticMapViewer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._num_channels = 2
        assert (
            self.env is not None
        ), "CompletedRateViewer needs a habitat.Env to be passed in!"
        self.completed_rate = self.env.get_metrics()["completed_rate"]
        x_pos = self._semantic_map.shape[1] * 0.25
        y_pos = self._semantic_map.shape[0] * 0.1
        self.completed_rate_text = self.topdown_axes.text(
            x_pos, y_pos, f"Completed Rate: {self.completed_rate:.4}", color="#73D0FF"
        )

    def __call__(self, event):
        super().__call__(event)
        self.completed_rate = self.env.get_metrics()["completed_rate"]
        self.completed_rate_text.set_text(f"Completed Rate: {self.completed_rate:.4}")
        self.fig.canvas.draw()


class ScanningRateViewer(SemanticMapViewer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._num_channels = 10
        assert (
            self.env is not None
        ), "ScanningRateViewer needs a habitat.Env to be passed in!"
        self.scanning_rate = self.env.get_metrics()["scanning_rate"]
        self.object_coverage = self.env.get_metrics()["object_coverage"]
        x_pos = self._semantic_map.shape[1] * 0.25
        y_pos = self._semantic_map.shape[0] * 0.1
        self.scanning_rate_text = self.topdown_axes.text(
            x_pos, y_pos, f"Scanning Rate: {self.scanning_rate:.4}", color="#73D0FF"
        )
        self.object_coverage_text = self.topdown_axes.text(
            x_pos, y_pos + 30, f"Object Coverage: {self.object_coverage:.4}", color="#95E6CB"
        )

    def __call__(self, event):
        super().__call__(event)
        self.scanning_rate = self.env.get_metrics()["scanning_rate"]
        self.object_coverage = self.env.get_metrics()["object_coverage"]
        self.scanning_rate_text.set_text(f"Scanning Rate: {self.scanning_rate:.4}")
        self.object_coverage_text.set_text(f"Object Coverage: {self.object_coverage:.4}")
        self.fig.canvas.draw()
