from contextlib import closing

import numpy as np
import cv2
import matplotlib.pyplot as plt

import habitat
from habitat.config.default import get_config

from habitat_scanbot2d.navigation_action import NavigationAction
from habitat_scanbot2d.test.test_sensors import create_env_with_one_episode
from habitat_scanbot2d.sensors import SemanticTopDownSensor
from habitat_scanbot2d import measures


def manually_explore_scene(env: habitat.Env):
    action_keys = {
        ord("w"): {"action": "MOVE_FORWARD", "action_args": None},
        ord("a"): {"action": "TURN_LEFT", "action_args": None},
        ord("d"): {"action": "TURN_RIGHT", "action_args": None},
    }

    obs = env.reset()
    obstacle_map = obs["global_semantic_map"][..., SemanticTopDownSensor.obstacle_channel]
    obstacle_with_rgb_img = concatenate_obstacle_with_rgb(obstacle_map, obs["rgb"])
    cv2.imshow("obstacle map", obstacle_with_rgb_img)
    key_code = cv2.waitKey(0)
    while key_code != 27:  # use Esc to exit
        action = action_keys.get(key_code, None)
        if action is not None:
            obs = env.step(action)
            obstacle_map = obs["global_semantic_map"][
                ..., SemanticTopDownSensor.obstacle_channel
            ]
            obstacle_with_rgb_img = concatenate_obstacle_with_rgb(
                obstacle_map, obs["rgb"]
            )
            cv2.imshow("obstacle map", obstacle_with_rgb_img)
        else:
            if key_code == ord("g"):
                goal_pose = obs["map_pose"]
        key_code = cv2.waitKey(0)

    return goal_pose


def test_reachable_goal(
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
        try:
            goal_pose = manually_explore_scene(env)
        except UnboundLocalError:
            print("Please press 'key g' for path goal!")
            raise

        cv2.destroyAllWindows()
        navigation_action = NavigationAction(
            config=config.TASK.ACTIONS.NAVIGATION, sim=env._sim, visualization=True
        )
        obs = navigation_action.step(
            task=env._task,  # type: ignore
            episode=env.current_episode,
            goal_position=goal_pose[:2].astype(np.int32),
        )


def test_unreachable_goal_with_cpu_observations(
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
    config.TASK.ACTIONS.NAVIGATION.VISUALIZATION = False
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


def test_unreachable_goal_with_cuda_observations(
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
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.TYPE = "SemanticTopDownCudaSensor"
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.MAP_CELL_SIZE = 0.1
    config.TASK.SEMANTIC_MAP_BUILDER.USE_LOCAL_REPRESENTATION = False
    config.TASK.SENSORS = ["MAP_POSE_SENSOR", "SEMANTIC_TOPDOWN_SENSOR"]
    config.TASK.ACTIONS.NAVIGATION.VISUALIZATION = False
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
    config.TASK.TYPE = "Scanning-v0"
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        obs = env.reset()
        navigation_action = NavigationAction(
            config=config.TASK.ACTIONS.NAVIGATION, sim=env._sim, visualization=False
        )
        unreachable_goal = np.array([0.0, -0.5])
        obs = navigation_action.step(
            task=env._task,  # type: ignore
            episode=env.current_episode,
            goal_position=unreachable_goal,
        )
        assert np.any(obs["map_pose"] != unreachable_goal)


def test_local_goal_with_cuda_observations(
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
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.TYPE = "SemanticTopDownCudaSensor"
    config.TASK.SENSORS = ["MAP_POSE_SENSOR", "SEMANTIC_TOPDOWN_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
    config.TASK.TYPE = "Scanning-v0"
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        obs = env.reset()
        navigation_action = NavigationAction(
            config=config.TASK.ACTIONS.NAVIGATION, sim=env._sim, visualization=True
        )
        local_goal = np.array([-1.0, -1.0])
        obs = navigation_action.step(
            task=env._task,  # type: ignore
            episode=env.current_episode,
            goal_position=local_goal,
        )
        assert np.any(obs["map_pose"] != local_goal)


def test_action_space():
    config = get_config("configs/scanning_task_gibson.yaml")

    with closing(create_env_with_one_episode(config)) as env:
        _ = env.reset()
        action_space = env.action_space["NAVIGATION"]
        print(action_space.sample())


def concatenate_obstacle_with_rgb(
    obstacle_map: np.ndarray, rgb: np.ndarray
) -> np.ndarray:
    channels = 3
    obstacle_map = (obstacle_map >= 1).astype(np.uint8) * 255
    obstacle_map = np.repeat(np.expand_dims(obstacle_map, axis=-1), channels, axis=-1)
    obstacle_map = cv2.resize(
        src=obstacle_map,
        dsize=None,
        fx=rgb.shape[1] / obstacle_map.shape[1],
        fy=rgb.shape[0] / obstacle_map.shape[0],
        interpolation=cv2.INTER_NEAREST,
    )
    result = np.concatenate((obstacle_map, rgb), axis=1)
    return result
