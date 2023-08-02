import os
import pytest
from contextlib import closing

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

import habitat
from habitat.config import get_config
from habitat_baselines.config.default import get_config as get_baseline_config
from habitat_baselines.slambased.mappers import DirectDepthMapper
from habitat_baselines.slambased.utils import generate_2dgrid
from habitat_scanbot2d.path_planners import DifferentiableStarPlanner
from habitat_scanbot2d.astar_path_finder import AStarPathFinder
from habitat_scanbot2d.scanning_task import ScanningEpisode
from habitat_scanbot2d.utils.semantic_map import construct_transformation_matrix
from habitat_scanbot2d.navigation_action import NavigationAction
from habitat_scanbot2d.test.test_sensors import create_env_with_one_episode


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
        obstacle_with_rgb_img = concatenate_obstacle_with_rgb(
            curr_obstacle_map.numpy(), obs["rgb"]
        )
        cv2.imshow(
            "Obstacle map",
            obstacle_with_rgb_img,
        )
        key_code = cv2.waitKey(0)
        while key_code != 27:  # use Esc to exit
            action = action_keys.get(key_code, None)
            if action is not None:
                print(action["action"])
                obs = env.step(action)
                agent_state = env._sim.get_agent_state()
                current_transformation = torch.from_numpy(
                    construct_transformation_matrix(agent_state)
                )

                obstacle_map = mapper(
                    torch.from_numpy(obs["depth"]).squeeze(),
                    initial_transformation.inverse() @ current_transformation,
                )
                curr_obstacle_map = torch.max(curr_obstacle_map, obstacle_map)
                # curr_obstacle_map = obstacle_map

            obstacle_with_rgb_img = concatenate_obstacle_with_rgb(
                curr_obstacle_map.numpy(), obs["rgb"]
            )
            cv2.imshow(
                "Obstacle map",
                obstacle_with_rgb_img,
            )
            key_code = cv2.waitKey(0)

        cv2.destroyAllWindows()

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

def test_astar_path_finder_in_small_grid():
    obstacle_map = np.zeros((5, 5), dtype=np.float32);
    obstacle_map[2, 2] = 1.0;
    obstacle_map[2, 3] = 1.0;
    obstacle_map[3, 2] = 1.0;

    start_point = np.array([1, 1])
    end_point = np.array([4, 3])

    astar_path_finder = AStarPathFinder()
    path = astar_path_finder.find(obstacle_map, start_point, end_point)
    print(path)

def test_astar_path_finder_with_manually_set_goal():
    config = get_config()
    baseline_config = get_baseline_config(opts=["BASE_TASK_CONFIG_PATH", None])
    mapper = DirectDepthMapper(
        camera_height=baseline_config.ORBSLAM2.CAMERA_HEIGHT,
        near_th=baseline_config.ORBSLAM2.D_OBSTACLE_MIN,
        far_th=baseline_config.ORBSLAM2.D_OBSTACLE_MAX,
        h_min=baseline_config.ORBSLAM2.H_OBSTACLE_MIN - 0.6,
        h_max=baseline_config.ORBSLAM2.H_OBSTACLE_MAX,
    )
    config.defrost()
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.freeze()

    action_keys = {
        ord("w"): {"action": "MOVE_FORWARD", "action_args": None},
        ord("a"): {"action": "TURN_LEFT", "action_args": None},
        ord("d"): {"action": "TURN_RIGHT", "action_args": None},
    }

    with closing(create_env_with_one_episode(config)) as env:
        obs = env.reset()
        agent_state = env._sim.get_agent_state()
        initial_transformation = torch.from_numpy(
            construct_transformation_matrix(agent_state)
        )

        obstacle_map = mapper(torch.from_numpy(obs["depth"]).squeeze(), torch.eye(4))
        curr_obstacle_map = obstacle_map
        obstacle_with_rgb_img = concatenate_obstacle_with_rgb(
            curr_obstacle_map.numpy(), obs["rgb"]
        )
        cv2.imshow(
            "Obstacle map",
            obstacle_with_rgb_img,
        )
        key_code = cv2.waitKey(0)
        while key_code != 27:  # use Esc to exit
            action = action_keys.get(key_code, None)
            if action is not None:
                print(action["action"])
                obs = env.step(action)
                agent_state = env._sim.get_agent_state()
                current_transformation = torch.from_numpy(
                    construct_transformation_matrix(agent_state)
                )

                obstacle_map = mapper(
                    torch.from_numpy(obs["depth"]).squeeze(),
                    initial_transformation.inverse() @ current_transformation,
                )
                curr_obstacle_map = torch.max(curr_obstacle_map, obstacle_map)
                # curr_obstacle_map = obstacle_map

            obstacle_with_rgb_img = concatenate_obstacle_with_rgb(
                curr_obstacle_map.numpy(), obs["rgb"]
            )
            cv2.imshow(
                "Obstacle map",
                obstacle_with_rgb_img,
            )
            key_code = cv2.waitKey(0)

        cv2.destroyAllWindows()

        plt.ion()
        start_point = np.array([200, 200])
        goal_point = np.array([170, 150])
        path_finder = AStarPathFinder()
        path = path_finder.find(
            curr_obstacle_map.numpy(),
            start_point,
            goal_point,
        )
        plt.ioff()
        plt.figure(num="obstacles with path")
        plt.imshow(curr_obstacle_map.numpy(), vmin=0.0, vmax=1.0)
        plt.plot(
            [path_point[1] for path_point in path],
            [path_point[0] for path_point in path],
            color="g",
        )
        plt.show()
