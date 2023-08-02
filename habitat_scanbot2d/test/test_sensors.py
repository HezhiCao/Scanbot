import os
from itertools import cycle
from contextlib import closing
from typing import cast
import pytest

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import habitat
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_scanbot2d.scanning_task import ScanningEpisode
from habitat_scanbot2d.sensors import SemanticTopDownSensor
from habitat_scanbot2d.utils.visualization import (
    visualize_semantic,
    construct_o3d_point_cloud,
    SemanticMapViewer,
)
from habitat_scanbot2d.utils.semantic_map import (
    extract_object_category_name,
    construct_transformation_matrix,
)


def create_env_with_full_episodes(config) -> habitat.Env:
    config.defrost()
    config.DATASET.CONTENT_SCENES = ["Arkansaw"]
    config.DATASET.DATA_PATH = (
        "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
    )
    config.freeze()

    env = habitat.Env(config=config, dataset=None)
    return env


def create_env_with_one_episode(config) -> habitat.Env:
    config.defrost()
    config.DATASET.CONTENT_SCENES = ["Albertville"]
    config.DATASET.DATA_PATH = (
        "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
    )
    config.freeze()

    env = habitat.Env(config=config, dataset=None)
    valid_start_position = [-1.31759, 0.0961462, -3.79719]
    start_rotation = [0, -0.542452, 0, -0.840087]
    # valid_start_position = [-1.36508,0.0110509,-0.391728]
    # start_rotation = [0, 1.22929e-05, 0, 1]

    env.episode_iterator = cycle(
        [
            ScanningEpisode(
                episode_id="0",
                scene_id=config.SIMULATOR.SCENE,
                start_position=valid_start_position,
                start_rotation=start_rotation,
            )
        ]
    )
    return env


@pytest.mark.filterwarnings(r"ignore:.*np\.:DeprecationWarning")
def test_semantic_top_down_sensor(
    semantic_topdown_sensor_config, map_pose_sensor_config
):
    config = semantic_topdown_sensor_config.clone()
    config.merge_from_other_cfg(map_pose_sensor_config)
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
        obs = env.reset()
        _ = SemanticMapViewer(
            obs,
            config.TASK.SEMANTIC_TOPDOWN_SENSOR.NUM_TOTAL_CHANNELS,
            object_category_name,
        )
        plt.show()


def test_point_cloud_sensor(point_cloud_sensor_config):
    config = point_cloud_sensor_config
    config.defrost()
    config.TASK.SENSORS = ["POINT_CLOUD_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        obs = env.reset()
        pcd = construct_o3d_point_cloud(obs["point_cloud"], obs["rgb"])
        o3d.visualization.draw_geometries([pcd])


@pytest.mark.filterwarnings(r"ignore:.*np\.:DeprecationWarning")
def test_point_cloud_integration(point_cloud_sensor_config):
    config = point_cloud_sensor_config
    config.defrost()
    config.TASK.SENSORS = ["POINT_CLOUD_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        init_obs = env.reset()
        init_pcd = construct_o3d_point_cloud(init_obs["point_cloud"], init_obs["rgb"])
        init_transform = construct_transformation_matrix(env._sim.get_agent_state())

        for _ in range(8):
            new_obs = env.step({"action": "TURN_LEFT", "action_args": None})
            new_pcd = construct_o3d_point_cloud(new_obs["point_cloud"], new_obs["rgb"])
            new_transform = construct_transformation_matrix(env._sim.get_agent_state())

            new_pcd = new_pcd.transform(np.linalg.inv(init_transform) @ new_transform)

            init_pcd.points.extend(new_pcd.points)  # type: o3d.geometry.PointCloud
            init_pcd.colors.extend(new_pcd.colors)  # type: o3d.geometry.PointCloud

    o3d.visualization.draw_geometries([init_pcd])


def test_category_semantic_sensor():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.CATEGORY_SEMANTIC_SENSOR = habitat.Config()
    config.TASK.CATEGORY_SEMANTIC_SENSOR.TYPE = "CategorySemanticSensor"
    config.TASK.SENSORS = ["CATEGORY_SEMANTIC_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "RGB_SENSOR", "SEMANTIC_SENSOR"]
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        init_obs = env.reset()
        init_semantic = init_obs["category_semantic"]
        visualize_semantic(init_semantic)


def test_category_semantic_sensor_before_stair():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.CATEGORY_SEMANTIC_SENSOR = habitat.Config()
    config.TASK.CATEGORY_SEMANTIC_SENSOR.TYPE = "CategorySemanticSensor"
    config.TASK.SENSORS = ["CATEGORY_SEMANTIC_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "RGB_SENSOR", "SEMANTIC_SENSOR"]
    config.freeze()

    valid_start_position = [-3.85287, 0.188068, -3.79332]
    start_rotation = [0, 0.99889, 0, 0.0470945]

    with habitat.Env(config) as env:
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
        init_obs = env.reset()
        init_semantic = init_obs["category_semantic"]
        visualize_semantic(init_semantic)


@pytest.mark.filterwarnings(r"ignore:.*np\.:DeprecationWarning")
def test_map_pose_sensor(map_pose_sensor_config):
    config = map_pose_sensor_config
    config.defrost()
    config.TASK.SENSORS = ["MAP_POSE_SENSOR"]
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        obs = env.reset()
        pose = obs["map_pose"]
        assert len(pose) == 4
        assert np.allclose(pose[:2], (200, 200))
        assert np.allclose(pose[2:], (-1.0, 0.0))

        obs = env.step({"action": "TURN_LEFT", "action_args": None})
        pose = obs["map_pose"]
        assert len(pose) == 4
        assert np.allclose(pose[:2], (200, 200))
        assert np.allclose(
            pose[2:],
            (
                -np.cos(np.deg2rad(config.SIMULATOR.TURN_ANGLE)),
                -np.sin(np.deg2rad(config.SIMULATOR.TURN_ANGLE)),
            ),
        )

        obs = env.step({"action": "MOVE_FORWARD", "action_args": None})
        pose = obs["map_pose"]
        assert len(pose) == 4
        assert np.allclose(
            pose[:2],
            np.round(
                np.array((200, 200))
                + np.array(pose[2:]) * config.SIMULATOR.FORWARD_STEP_SIZE / 0.1
            ),
        )
        assert np.allclose(
            pose[2:],
            (
                -np.cos(np.deg2rad(config.SIMULATOR.TURN_ANGLE)),
                -np.sin(np.deg2rad(config.SIMULATOR.TURN_ANGLE)),
            ),
        )


def test_semantic_top_down_quality_channel(semantic_topdown_sensor_config):
    config = semantic_topdown_sensor_config
    config.defrost()
    config.TASK.SENSORS = [
        "SEMANTIC_TOPDOWN_SENSOR",
    ]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        obs = env.reset()
        scanning_quality_map = obs[SemanticTopDownSensor.uuid][
            ...,
            SemanticTopDownSensor.quality_channel : SemanticTopDownSensor.quality_channel
            + 8,
        ]
        assert np.max(scanning_quality_map) == 1.0
        assert np.min(scanning_quality_map) == 0.0

        scanning_distance_histogram = np.histogram(scanning_quality_map, bins=11)[0]
        for i, count in enumerate(scanning_distance_histogram):
            if i % 2 == 0:
                assert count > 0
            else:
                assert count == 0
