from contextlib import closing

import torch
from torch.nn import functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

import habitat
from habitat_baselines.config.default import get_config as get_baseline_config
from habitat_baselines.slambased.mappers import DirectDepthMapper
from habitat_baselines.slambased.utils import generate_2dgrid

from habitat_scanbot2d.sensors import MapPoseSensor
from habitat_scanbot2d.path_planners import DifferentiableStarPlanner
from habitat_scanbot2d.primitive_action_planner import PrimitiveActionPlanner
from habitat_scanbot2d.utils.semantic_map import construct_transformation_matrix
from habitat_scanbot2d.utils.visualization import construct_fancy_arrow
from habitat_scanbot2d.test.test_sensors import create_env_with_one_episode


def manually_explore_scene(env: habitat.Env, mapper: DirectDepthMapper):
    action_keys = {
        ord("w"): {"action": "MOVE_FORWARD", "action_args": None},
        ord("a"): {"action": "TURN_LEFT", "action_args": None},
        ord("d"): {"action": "TURN_RIGHT", "action_args": None},
    }

    obs = env.reset()
    init_state = env._sim.get_agent_state()
    init_transformation = torch.from_numpy(construct_transformation_matrix(init_state))
    obstacles_map = mapper(torch.from_numpy(obs["depth"]).squeeze(), torch.eye(4))
    obstacles_with_rgb_img = concatenate_obstacle_with_rgb(
        obstacles_map.numpy(), obs["rgb"]
    )
    cv2.imshow("obstacle map", obstacles_with_rgb_img)
    key_code = cv2.waitKey(0)
    while key_code != 27:  # use Esc to exit
        action = action_keys.get(key_code, None)
        if action is not None:
            obs = env.step(action)
            curr_transformation = torch.from_numpy(
                construct_transformation_matrix(env._sim.get_agent_state())
            )
            curr_obstacles = mapper(
                torch.from_numpy(obs["depth"]).squeeze(),
                init_transformation.inverse() @ curr_transformation,
            )
            obstacles_map = torch.max(obstacles_map, curr_obstacles)
            obstacles_with_rgb_img = concatenate_obstacle_with_rgb(
                obstacles_map.numpy(), obs["rgb"]
            )
            cv2.imshow("obstacle map", obstacles_with_rgb_img)
        else:
            if key_code == ord("s"):
                start_pose = obs["map_pose"]
                start_state = env._sim.get_agent_state()
            if key_code == ord("g"):
                goal_pose = obs["map_pose"]
        key_code = cv2.waitKey(0)

    return obstacles_map, start_pose, start_state, goal_pose


def test_primitive_action_planner(map_pose_sensor_config, primitive_action_planner_config):
    config = map_pose_sensor_config.clone()
    config.merge_from_other_cfg(primitive_action_planner_config)
    baseline_config = get_baseline_config(opts=["BASE_TASK_CONFIG_PATH", None])
    config.defrost()
    config.TASK.SENSORS = ["MAP_POSE_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        mapper = DirectDepthMapper(
            camera_height=baseline_config.ORBSLAM2.CAMERA_HEIGHT,
            near_th=baseline_config.ORBSLAM2.D_OBSTACLE_MIN,
            far_th=baseline_config.ORBSLAM2.D_OBSTACLE_MAX,
            h_min=baseline_config.ORBSLAM2.H_OBSTACLE_MIN - 0.6,
            h_max=baseline_config.ORBSLAM2.H_OBSTACLE_MAX,
        )

        try:
            obstacles_map, start_pose, start_state, goal_pose = manually_explore_scene(
                env, mapper
            )
        except UnboundLocalError:
            print("Please press 'key s' and 'key g' for path start and path goal!")
            raise

        grid2d = generate_2dgrid(
            obstacles_map.size()[0], obstacles_map.size()[1], centered=False
        )
        start_map = torch.zeros_like(obstacles_map.unsqueeze(0).unsqueeze(0))
        goal_map = torch.zeros_like(obstacles_map.unsqueeze(0).unsqueeze(0))
        start_map[0, 0, int(start_pose[0]), int(start_pose[1])] = 1.0
        goal_map[0, 0, int(goal_pose[0]), int(goal_pose[1])] = 1.0

        cv2.destroyAllWindows()

        plt.ion()
        planner = DifferentiableStarPlanner(preprocess=True, visualize=False)
        path, cost = planner(
            obstacles=rawmap2_planner_ready(
                obstacles_map.unsqueeze(0).unsqueeze(0), start_map, goal_map
            ),
            coords=grid2d,
            start_map=goal_map,
            goal_map=start_map,
        )

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(obstacles_map, vmin=0.0, vmax=1.0)
        ax.plot([point[1] for point in path], [point[0] for point in path], color="g")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        agent_waypoints = []
        actions = []
        agent = PrimitiveActionPlanner(config.TASK.PRIMITIVE_ACTION_PLANNER, path)
        env._sim.set_agent_state(start_state.position, start_state.rotation)

        agent_map_waypoint = start_pose
        arrow = construct_fancy_arrow(start_pose)
        arrow_patch = ax.add_patch(arrow)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.05)
        agent_waypoints.append(agent_map_waypoint)
        while True:
            action = agent.step(agent_map_waypoint)
            obs = env.step(action)
            arrow_patch.remove()
            arrow = construct_fancy_arrow(obs[MapPoseSensor.uuid])
            arrow_patch = ax.add_patch(arrow)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

            agent_map_waypoint = obs[MapPoseSensor.uuid]
            agent_waypoints.append(agent_map_waypoint)
            actions.append(action)
            if action["action"] == "STOP":
                break

        ax.plot(
            [point[1] for point in agent_waypoints],
            [point[0] for point in agent_waypoints],
            "r-",
        )
        fig.canvas.draw_idle()
        plt.ioff()
        plt.show()


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


def rawmap2_planner_ready(rawmap, start_map, goal_map):
    obstacle_th = 320
    map1 = (rawmap / float(obstacle_th)) ** 2
    map1 = (
        torch.clamp(map1, min=0, max=1.0)
        - start_map
        - F.max_pool2d(goal_map, 3, stride=1, padding=1)
    )
    return torch.relu(map1)
