from contextlib import closing
import habitat
import matplotlib.pyplot as plt
import cv2
import numpy as np

from itertools import cycle

from habitat_scanbot2d.local_object_voxelizer import LocalObjectVoxelizer
from habitat_scanbot2d.test.test_sensors import create_env_with_one_episode
import os
import pytest

from habitat_scanbot2d.voxel_builder_task import BuildVoxelEpisode
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.nav import NavigationGoal
from typing import Dict, Any
from habitat.config.default import Config, get_config

from habitat_scanbot2d.local_scanning_action import LocalScanningAction


def manually_explore_scene(env: habitat.Env):
    action_keys = {
        ord("w"): {"action": "MOVE_FORWARD", "action_args": None},
        ord("a"): {"action": "TURN_LEFT", "action_args": None},
        ord("d"): {"action": "TURN_RIGHT", "action_args": None},
    }
    obs = env.reset()
    cv2.imshow("RGB", obs["rgb"])
    key_code = cv2.waitKey(0)
    while key_code != 27:  # use Esc to exit
        action = action_keys.get(key_code, None)
        if action is not None:
            obs = env.step(action)
            cv2.imshow("RGB", obs["rgb"])
        key_code = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return obs


def test_local_object_voxelizer(point_cloud_sensor_config):
    config = point_cloud_sensor_config
    config.defrost()
    config.TASK.SENSORS = ["POINT_CLOUD_SENSOR"]
    config.TASK.VOXEL_BUILDER = Config()
    config.TASK.VOXEL_BUILDER.OBJECT_NUMBER = 3
    config.TASK.VOXEL_BUILDER.INITIAL_CAMERA_HEIGHT = 1.25
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:

        obs = manually_explore_scene(env)
        local_object_voxelizer = LocalObjectVoxelizer(
            config.TASK.VOXEL_BUILDER, env._sim
        )
        voxel = local_object_voxelizer.voxelize_extracted_object(
            obs, 3, visualization=True
        )


def test_local_scanning_action(
    point_nav_config,
    local_scanning_action_config,
):
    # **note that the object voxel is not up to date since the ** voxelize_and_get_object_position **
    # function in voxel_builder has not been called
    config = point_nav_config.clone()
    config.merge_from_other_cfg(local_scanning_action_config)
    config.defrost()
    config.TASK.SEMANTIC_MAP_BUILDER.USE_LOCAL_REPRESENTATION = False
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.MAP_CELL_SIZE = 0.1
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.MAP_CELL_SIZE = 0.1
    config.TASK.SENSORS = [
        "MAP_POSE_SENSOR",
        "SEMANTIC_TOPDOWN_SENSOR",
        "POINT_CLOUD_SENSOR",
    ]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.TASK.TYPE = "BuildVoxel-v0"
    config.freeze()

    with closing(create_navigation_env_with_one_episode(config)) as env:
        view_point_probability = np.zeros(
            8 * config.TASK.ACTIONS.LOCAL_SCANNING.OBJECT_NUMBER
        )
        # view_point_probability[2] = 1
        view_point_probability[1] = 1
        action = convert_actor_output_to_local_scanning_action(view_point_probability)
        obs = env.reset()
        obs = env.step(action)


def convert_actor_output_to_local_scanning_action(
    view_point_probability: int,
) -> Dict[str, Any]:
    r"""Construct a navigation action from the output of the actor

    :param actor_output: (2,) float tensor
    :return: Action dict with "action" and "action_args"
    """
    # goal_position = np.clip(actor_output, a_min=-1.0, a_max=1.0)
    return {
        "action": "LOCAL_SCANNING",
        "action_args": {"view_point_probability": view_point_probability},
    }


def convert_actor_output_to_navigation_action(
    actor_output: np.ndarray,
) -> Dict[str, Any]:
    r"""Construct a navigation action from the output of the actor

    :param actor_output: (2,) float tensor
    :return: Action dict with "action" and "action_args"
    """
    goal_position = np.clip(actor_output, a_min=-1.0, a_max=1.0)
    return {
        "action": "NAVIGATION",
        "action_args": {"goal_position": goal_position},
    }


def create_navigation_env_with_one_episode(config) -> habitat.Env:
    config.defrost()
    config.DATASET.CONTENT_SCENES = ["Albertville"]
    config.DATASET.DATA_PATH = (
        "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
    )
    config.freeze()

    env = habitat.Env(config=config, dataset=None)
    valid_start_position = [-1.31759, 0.0961462, -3.79719]
    start_rotation = [0, -0.542452, 0, -0.840087]
    [-0.138299, -0.208028]
    goal_position = [NavigationGoal(position=[-0.08, -0.08], radius=None)]

    env.episode_iterator = cycle(
        [
            NavigationEpisode(
                episode_id="0",
                scene_id=config.SIMULATOR.SCENE,
                start_position=valid_start_position,
                start_rotation=start_rotation,
                goals=goal_position,
            )
        ]
    )
    return env


def check_data_folder(config: Config):
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")


@pytest.fixture
def point_nav_config(
    map_pose_sensor_config,
    navigation_action_config,
    semantic_map_builder_config,
    semantic_topdown_sensor_config,
    point_cloud_sensor_config,
    voxel_builder_config,
):
    config = map_pose_sensor_config.clone()
    config.merge_from_other_cfg(semantic_topdown_sensor_config)
    config.merge_from_other_cfg(semantic_map_builder_config)
    config.merge_from_other_cfg(point_cloud_sensor_config)
    config.merge_from_other_cfg(navigation_action_config)
    config.merge_from_other_cfg(voxel_builder_config)
    config.freeze()
    return config


@pytest.fixture(scope="session")
def local_scanning_action_config(primitive_action_planner_config):
    config = get_config()
    check_data_folder(config)
    config.defrost()
    config.TASK.ACTIONS.LOCAL_SCANNING = Config()
    local_scanning_config = config.TASK.ACTIONS.LOCAL_SCANNING
    local_scanning_config.TYPE = "LocalScanningAction"
    local_scanning_config.VISUALIZATION = True
    local_scanning_config.BASELINE = Config()
    local_scanning_config.VIEW_POINT_SHIFT = [0.5, 0, 0.5]
    local_scanning_config.OBJECT_NUMBER = 13
    local_scanning_config.PRIMITIVE_ACTION_PLANNER = (
        primitive_action_planner_config.TASK.PRIMITIVE_ACTION_PLANNER
    )
    config.TASK.POSSIBLE_ACTIONS = [
        "STOP",
        "MOVE_FORWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "NAVIGATION",
        "LOCAL_SCANNING",
    ]
    config.freeze()
    return config


@pytest.fixture(scope="session")
def voxel_builder_config():
    config = get_config()
    check_data_folder(config)
    config.defrost()
    config.TASK.VOXEL_BUILDER = Config()
    config.TASK.VOXEL_BUILDER.SCENE_SIZE = [5, 2.5, 5]
    config.TASK.VOXEL_BUILDER.SCENE_VOXEL_SHIFT = 0
    config.TASK.VOXEL_BUILDER.OBJETS_VOXEL_NUMBER_PER_LENGTH = 32
    config.TASK.VOXEL_BUILDER.SCENE_VOXEL_NUMBER_PER_LENGTH = 64
    config.TASK.VOXEL_BUILDER.INITIAL_CAMERA_HEIGHT = 1.25
    config.TASK.VOXEL_BUILDER.H_MIN_THRESHOLD = 0
    config.TASK.VOXEL_BUILDER.H_MAX_THRESHOLD = 3
    config.TASK.VOXEL_BUILDER.SAVE_SCENE_DATA = True
    config.TASK.VOXEL_BUILDER.SAVE_DIR = "./scene_data"
    config.freeze()
    return config
