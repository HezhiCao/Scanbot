from contextlib import closing
import pytest
import torch
import matplotlib.pyplot as plt
import open3d as o3d

import habitat
from habitat_scanbot2d.test.test_sensors import create_env_with_one_episode
from habitat_scanbot2d.sensors import SemanticTopDownSensor
from habitat_scanbot2d.cuda_sensors import SemanticTopDownCudaSensor
from habitat_scanbot2d.utils.visualization import (
    SemanticMapViewer,
    construct_o3d_point_cloud,
    visualize_semantic,
)
from habitat_scanbot2d.utils.semantic_map import extract_object_category_name


def test_point_cloud_cuda_sensor(point_cloud_sensor_config):
    config = point_cloud_sensor_config
    config.defrost()
    config.TASK.POINT_CLOUD_SENSOR.TYPE = "PointCloudCudaSensor"
    config.TASK.SENSORS = ["POINT_CLOUD_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        obs = env.reset()
        pcd = construct_o3d_point_cloud(
            obs["point_cloud"].cpu().numpy(), obs["rgb"].cpu().numpy()
        )
        o3d.visualization.draw_geometries([pcd])


def test_category_semantic_cuda_sensor(category_semantic_sensor_config):
    config = category_semantic_sensor_config
    config.defrost()
    config.TASK.CATEGORY_SEMANTIC_SENSOR.TYPE = "CategorySemanticCudaSensor"
    config.TASK.SENSORS = ["CATEGORY_SEMANTIC_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "RGB_SENSOR", "SEMANTIC_SENSOR"]
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        init_obs = env.reset()
        init_semantic = init_obs["category_semantic_cuda"].cpu().numpy()
        visualize_semantic(init_semantic)


@pytest.mark.filterwarnings(r"ignore:.*np\.:DeprecationWarning")
def test_semantic_top_down_cuda_sensor(
    semantic_topdown_sensor_config, map_pose_sensor_config
):
    config = semantic_topdown_sensor_config.clone()
    config.merge_from_other_cfg(map_pose_sensor_config)
    config.defrost()
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.TYPE = "SemanticTopDownCudaSensor"
    config.TASK.SENSORS = [
        "SEMANTIC_TOPDOWN_SENSOR",
        "MAP_POSE_SENSOR",
    ]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        object_category_name = extract_object_category_name(
            env._sim.semantic_annotations()  # type: ignore
        )
        obs = env.reset()
        obs = { k: v.cpu().numpy() for k, v in obs.items() if isinstance(v, torch.Tensor)}
        _ = SemanticMapViewer(
            obs,
            config.TASK.SEMANTIC_TOPDOWN_SENSOR.NUM_TOTAL_CHANNELS,
            object_category_name,
        )
        plt.show()

def test_semantic_top_down_cuda_quality_channel(semantic_topdown_sensor_config):
    config = semantic_topdown_sensor_config
    config.defrost()
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.TYPE = "SemanticTopDownCudaSensor"
    config.TASK.SENSORS = [
        "SEMANTIC_TOPDOWN_SENSOR",
    ]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
    config.freeze()

    with closing(create_env_with_one_episode(config)) as env:
        obs = env.reset()
        scanning_quality_map = obs[SemanticTopDownCudaSensor.uuid][
            ...,
            SemanticTopDownSensor.quality_channel : SemanticTopDownSensor.quality_channel
            + 8,
        ]
        assert torch.max(scanning_quality_map) == 1.0
        assert torch.min(scanning_quality_map) == 0.0

        scanning_distance_histogram = torch.histc(scanning_quality_map, bins=11)
        for i, count in enumerate(scanning_distance_histogram.cpu().numpy()):
            if i % 2 == 0:
                assert count > 0
            else:
                assert count == 0
