from contextlib import closing
import habitat
import habitat_sim
import matplotlib.pyplot as plt
import numpy as np

from habitat.config.default import Config

from habitat_scanbot2d.cuda_sensors import SemanticTopDownCudaSensor
from habitat_scanbot2d.semantic_map_builder import SemanticMapBuilder
from habitat_scanbot2d.utils.semantic_map import extract_object_category_name
from habitat_scanbot2d.utils.visualization import SemanticMapViewer
from habitat_scanbot2d.test.test_sensors import (
    create_env_with_one_episode,
    create_env_with_full_episodes,
)
from habitat_scanbot2d.test.test_mp3d import create_mp3d_env_with_one_episode
from habitat_scanbot2d.voxel_builder import VoxelBuilder

from habitat_scanbot2d.navigation_action import NavigationAction


def test_semantic_map_builder_with_cpu_observations(
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
        map_builder.update(obs)
        _ = SemanticMapViewer(
            obs,
            config.TASK.SEMANTIC_TOPDOWN_SENSOR.NUM_TOTAL_CHANNELS,
            object_category_name,
            None,
            env,
            map_builder,
        )
        plt.show()


def test_semantic_map_builder_with_cuda_observations(
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
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.TYPE = "SemanticTopDownCudaSensor"
    config.TASK.POSSIBLE_ACTIONS = [
        "STOP",
        "MOVE_FORWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "LOOK_UP",
        "LOOK_DOWN",
    ]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.ACTION_SPACE_CONFIG = "v1"
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
    config.freeze()

    def normalize_obstacle_fn(obstacle_map: np.ndarray) -> np.ndarray:
        return NavigationAction.normalize_obstacle_map(
            obstacle_map, 10
        ).numpy()

                                                  # start_position=[13.5309,0.120891,4.53464],
                                                  # start_rotation=[0, 0.0549625, 0, 0.998488]
                                                  # 5ZKStnWn8Zo
    with closing(create_mp3d_env_with_one_episode(config,
                                                  scene_id="UwV83HsGsw3",
                                                  start_position=[-15.5335,0.0645995,1.77988],
                                                  start_rotation=[0,-0.999229,0,0.0392723]
                                                  )) as env:
        object_category_name = config.TASK.SEMANTIC_TOPDOWN_SENSOR.SEMANTIC_CHANNEL_CATEGORIES
        map_builder = SemanticMapBuilder(config.TASK.SEMANTIC_MAP_BUILDER)
        obs = env.reset()
        map_builder.update(obs)
        _ = SemanticMapViewer(
            obs,
            config.TASK.SEMANTIC_TOPDOWN_SENSOR.NUM_TOTAL_CHANNELS,
            object_category_name,
            normalize_obstacle_fn,
            env,
            map_builder,
        )
        plt.show()
