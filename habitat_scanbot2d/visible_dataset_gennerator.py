from contextlib import closing
import habitat
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

from habitat.config.default import Config

from habitat_scanbot2d.semantic_map_builder import SemanticMapBuilder
from habitat_scanbot2d.utils.semantic_map import extract_object_category_name
from habitat_scanbot2d.utils.visualization import SemanticTopDownViewer
from habitat_scanbot2d.test.test_sensors import create_env_with_one_episode
from habitat_scanbot2d.voxel_builder import VoxelBuilder
from habitat_scanbot2d.navigation_action import NavigationAction
import os
import pytest

from habitat.config import get_config
from habitat_baselines.config.default import get_config as get_baseline_config
from habitat_baselines.slambased.mappers import DirectDepthMapper
from habitat_baselines.slambased.utils import generate_2dgrid
from habitat_scanbot2d.path_planners import DifferentiableStarPlanner
from habitat_scanbot2d.astar_path_finder import AStarPathFinder
from habitat_scanbot2d.scanning_task import ScanningEpisode
from habitat_scanbot2d.utils.semantic_map import construct_transformation_matrix

def test_semantic_map_builder(
    semantic_topdown_sensor_config, map_pose_sensor_config, semantic_map_builder_config
):
    config = semantic_topdown_sensor_config.clone()
    config.merge_from_other_cfg(map_pose_sensor_config)
    config.merge_from_other_cfg(semantic_map_builder_config)
    config.defrost()
    config.TASK.SENSORS = [
        "SEMANTIC_TOPDOWN_SENSOR",
        "MAP_POSE_SENSOR",
    ]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        object_category_name = extract_object_category_name(
            env._sim.semantic_annotations()  # type: ignore
        )
        map_builder = SemanticMapBuilder(config.TASK.SEMANTIC_MAP_BUILDER)
        obs = env.reset()
        _ = SemanticTopDownViewer(
            obs,
            config.TASK.SEMANTIC_TOPDOWN_SENSOR.NUM_TOTAL_CHANNELS,
            object_category_name,
            env,
            map_builder,
        )
        plt.show()


def test_voxel_builder(point_cloud_sensor_config):
    config = point_cloud_sensor_config.clone()
    config.defrost()
    config.TASK.VOXEL_BUILDER = Config()
    config.TASK.VOXEL_BUILDER.OBJECT_NUMBER = 3
    config.TASK.SENSORS = ["POINT_CLOUD_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = [
        "DEPTH_SENSOR",
        "SEMANTIC_SENSOR",
        "RGB_SENSOR",
    ]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.freeze()
    with closing(create_env_with_one_episode(config)) as env:
        action_keys = {
            ord("w"): {"action": "MOVE_FORWARD", "action_args": None},
            ord("d"): {"action": "TURN_RIGHT", "action_args": None},
            ord("a"): {"action": "TURN_LEFT", "action_args": None},
        }

        obs = env.reset()
        voxel_builder = VoxelBuilder(config, env._sim, obs)
        cv2.imshow("rgb map", obs["rgb"])
        key_code = cv2.waitKey(0)
        while key_code != 27:  # use Esc to exit
            action = action_keys.get(key_code, None)
            if action is not None:
                obs = env.step(action)
                cv2.imshow("rgb map", obs["rgb"])
                voxel_builder.update(obs)

            key_code = cv2.waitKey(0)
        voxel_builder.visualize()


def test_unreachable_goal(
    map_pose_sensor_config,
    navigation_action_config,
    semantic_map_builder_config,
    semantic_topdown_sensor_config,
):
    config = map_pose_sensor_config.clone()
    config.merge_from_other_cfg(semantic_topdown_sensor_config)
    config.merge_from_other_cfg(navigation_action_config)
    config.merge_from_other_cfg(semantic_map_builder_config)
    config.defrost()
    config.TASK.SENSORS = ["MAP_POSE_SENSOR", "SEMANTIC_TOPDOWN_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.TASK.TYPE = "Scanning-v0"
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        obs = env.reset()
        navigation_action = NavigationAction(
            config=config.TASK.ACTIONS.NAVIGATION, sim=env._sim, visualization=True
        )
        unreachable_goal = np.array([0.0, -0.5])
        obs = navigation_action.step(
            task=env._task,  # type: ignore
            episode=env.current_episode,
            goal_position=unreachable_goal,
        )
        assert np.any(obs["map_pose"] != unreachable_goal)

def test_path_planner():
    config = get_config()
    baseline_config = get_baseline_config(opts=["BASE_TASK_CONFIG_PATH", None])
    mapper = DirectDepthMapper(
        camera_height=baseline_config.ORBSLAM2.CAMERA_HEIGHT,
        near_th=baseline_config.ORBSLAM2.D_OBSTACLE_MIN,
        far_th=baseline_config.ORBSLAM2.D_OBSTACLE_MAX,
        h_min=baseline_config.ORBSLAM2.H_OBSTACLE_MIN - 0.6,
        h_max=baseline_config.ORBSLAM2.H_OBSTACLE_MAX,
    )
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.DATASET.CONTENT_SCENES = ["Albertville"]
    config.DATASET.DATA_PATH = (
        "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
    )
    config.freeze()

    action_keys = {
        ord("w"): {"action": "MOVE_FORWARD", "action_args": None},
        ord("a"): {"action": "TURN_LEFT", "action_args": None},
        ord("d"): {"action": "TURN_RIGHT", "action_args": None},
    }

    with habitat.Env(config=config, dataset=None) as env:
        valid_start_position = [-1.31759, 0.0961462, -3.79719]
        # start_rotation = [0., 0., 0., 1.]
        start_rotation = [0, -0.542452, 0, -0.840087]

        env.episode_iterator = iter(
            [
                ScanningEpisode(
                    episode_id="0",
                    scene_id=config.SIMULATOR.SCENE,
                    start_position=valid_start_position,
                    start_rotation=start_rotation,
                )
            ]
        )
        obs = env.reset()
        agent_state = env._sim.get_agent_state()
        initial_transformation = torch.from_numpy(
            construct_transformation_matrix(agent_state)
        )

        obstacle_map = mapper(torch.from_numpy(obs["depth"]).squeeze(), torch.eye(4))
        curr_obstacle_map = obstacle_map


        coordinatesGrid = generate_2dgrid(
            curr_obstacle_map.shape[0], curr_obstacle_map.shape[1], False
        ).to("cpu")

        plt.ion()
        curr_obstacle_map.unsqueeze_(0).unsqueeze_(0)
        start_map = torch.zeros_like(curr_obstacle_map).to("cpu")
        start_map[0, 0, 200, 200] = 1.0
        goal_map = torch.zeros_like(curr_obstacle_map).to("cpu")
        goal_map[
            0,
            0,
            170,
            150,
        ] = 1.0
        planner = DifferentiableStarPlanner(preprocess=True, visualize=True)
        path, _ = planner(
            NavigationAction.normalize_obstacle_map(
                curr_obstacle_map, start_map, goal_map
            ).to("cpu"),
            coordinatesGrid.to("cpu"),
            goal_map.to("cpu"),
            start_map.to("cpu"),
        )
        plt.ioff()
        plt.figure(num="obstacles with path")
        plt.imshow(curr_obstacle_map.squeeze(), vmin=0.0, vmax=1.0)
        plt.plot(
            [path_point[1].item() for path_point in path],
            [path_point[0].item() for path_point in path],
            color="g",
        )
        plt.show()

def concatenate_obstacle_with_rgb(
    obstacle_map: np.ndarray, rgb_img: np.ndarray
) -> np.ndarray:
    obstacle_threshold = 1.0
    obstacle_img = (obstacle_map >= obstacle_threshold).astype(np.uint8) * 255
    obstacle_img = np.repeat(np.expand_dims(obstacle_img, axis=-1), 3, axis=-1)
    resize_scale = rgb_img.shape[0] / obstacle_img.shape[0]
    result = cv2.resize(
        obstacle_img,
        None,
        fx=resize_scale,
        fy=resize_scale,
        interpolation=cv2.INTER_NEAREST,
    )
    result = np.concatenate((result, rgb_img), axis=1)
    return result
