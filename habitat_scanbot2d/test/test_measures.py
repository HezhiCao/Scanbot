from contextlib import closing
import pytest
import numpy as np
import matplotlib.pyplot as plt

import habitat
from habitat.config.default import Config

from habitat_scanbot2d.measures import (
    CompletedArea,
    CompletedRate,
    ScanningQuality,
    ScanningSuccess,
)
from habitat_scanbot2d.test.test_sensors import create_env_with_one_episode
from habitat_scanbot2d.utils.visualization import (
    CompletedRateViewer,
    ScanningRateViewer,
    SemanticMapViewer,
)
from habitat_scanbot2d.utils.semantic_map import extract_object_category_name
from habitat_scanbot2d.scanning_task import ScanningEpisode
from habitat_scanbot2d.semantic_map_builder import SemanticMapBuilder
from habitat_scanbot2d.test.test_mp3d import create_mp3d_env_with_one_episode

# pyright: reportGeneralTypeIssues=false
# pylint: disable=no-member


@pytest.fixture(scope="class")
def completed_area_measure_config(
    request, semantic_topdown_sensor_config, semantic_map_builder_config
):
    request.cls.config = semantic_topdown_sensor_config.clone()
    request.cls.config.merge_from_other_cfg(semantic_map_builder_config)
    request.cls.config.defrost()
    request.cls.config.TASK.TYPE = "Scanning-v0"
    request.cls.config.TASK.SENSORS = ["SEMANTIC_TOPDOWN_SENSOR"]
    request.cls.config.TASK.MEASUREMENTS = ["COMPLETED_AREA"]
    request.cls.config.TASK.COMPLETED_AREA = Config()
    request.cls.config.TASK.COMPLETED_AREA.TYPE = "CompletedArea"
    request.cls.config.SIMULATOR.AGENT_0.SENSORS = [
        "DEPTH_SENSOR",
        "SEMANTIC_SENSOR",
        "RGB_SENSOR",
    ]
    request.cls.config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    request.cls.config.freeze()
    return request.cls.config


@pytest.mark.usefixtures("completed_area_measure_config")
class TestCompletedAreaMeasure:
    def test_semantic_topdown_sensor_dependence(self):
        config = self.config.clone()
        config.defrost()
        config.TASK.TYPE = "Nav-v0"
        config.TASK.SENSORS = []
        config.freeze()
        with pytest.raises(AssertionError, match="requires a SemanticTopDownSensor"):
            env = create_env_with_one_episode(config)
            env.reset()
            env.close()

    def test_area_increment(self):
        with closing(create_env_with_one_episode(self.config)) as env:
            env.reset()
            metrics = env.get_metrics()
            assert np.allclose(metrics[CompletedArea.uuid], 11.17)

            env.step({"action": "MOVE_FORWARD", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[CompletedArea.uuid], 12.34)

            env.step({"action": "TURN_LEFT", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[CompletedArea.uuid], 13.58)

            env.step({"action": "TURN_RIGHT", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[CompletedArea.uuid], 13.58)

            env.step({"action": "MOVE_FORWARD", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[CompletedArea.uuid], 14.85)


@pytest.fixture(scope="class")
def completed_rate_measure_config(request, completed_area_measure_config):
    request.cls.config = completed_area_measure_config
    request.cls.config.defrost()
    request.cls.config.TASK.MEASUREMENTS = ["COMPLETED_AREA", "COMPLETED_RATE"]
    request.cls.config.TASK.COMPLETED_RATE = Config()
    request.cls.config.TASK.COMPLETED_RATE.TYPE = "CompletedRate"
    request.cls.config.freeze()
    return request.cls.config


@pytest.mark.usefixtures("completed_rate_measure_config")
class TestCompletedRateMeasure:
    def test_completed_area_measure_dependence_listed(self):
        config = self.config.clone()
        config.defrost()
        config.TASK.MEASUREMENTS = ["COMPLETED_RATE"]
        config.freeze()

        with pytest.raises(AssertionError, match="listed in the measures list"):
            env = create_env_with_one_episode(config)
            env.reset()
            env.close()

    def test_completed_area_measure_dependence_order(self):
        config = self.config.clone()
        config.defrost()
        config.TASK.MEASUREMENTS = ["COMPLETED_RATE", "COMPLETED_AREA"]
        config.freeze()

        with pytest.raises(AssertionError, match="requires be listed after"):
            env = create_env_with_one_episode(config)
            env.reset()
            env.close()

    def test_rate_increment(self):
        with closing(create_env_with_one_episode(self.config)) as env:
            env.reset()
            metrics = env.get_metrics()
            assert np.allclose(metrics[CompletedRate.uuid], 0.1627568)

            env.step({"action": "MOVE_FORWARD", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[CompletedRate.uuid], 0.1798048)

            env.step({"action": "TURN_LEFT", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[CompletedRate.uuid], 0.1978727)

            env.step({"action": "TURN_RIGHT", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[CompletedRate.uuid], 0.1978727)

            env.step({"action": "MOVE_FORWARD", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[CompletedRate.uuid], 0.2163777)

    def test_completed_rate_for_finished_scene(self, map_pose_sensor_config):
        config = map_pose_sensor_config.clone()
        config.merge_from_other_cfg(self.config)
        config.defrost()
        config.TASK.SENSORS = [
            "SEMANTIC_TOPDOWN_SENSOR",
            "MAP_POSE_SENSOR",
        ]
        config.freeze()

        with closing(create_env_with_one_episode(config)) as env:
            object_category_name = extract_object_category_name(
                env._sim.semantic_annotations()
            )
            obs = env.reset()
            _ = CompletedRateViewer(
                obs,
                config.TASK.SEMANTIC_TOPDOWN_SENSOR.NUM_TOTAL_CHANNELS,
                object_category_name,
                env,
                env.task.semantic_map_builder,
            )
            plt.show()

    def test_completed_rate_for_finished_scene_with_cuda_observations(
        self, map_pose_sensor_config
    ):
        config = map_pose_sensor_config.clone()
        config.merge_from_other_cfg(self.config)
        config.defrost()
        config.TASK.SENSORS = [
            "SEMANTIC_TOPDOWN_SENSOR",
            "MAP_POSE_SENSOR",
        ]
        config.TASK.SEMANTIC_TOPDOWN_SENSOR.TYPE = "SemanticTopDownCudaSensor"
        config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
        config.freeze()

        with closing(create_env_with_one_episode(config)) as env:
            object_category_name = extract_object_category_name(
                env._sim.semantic_annotations()
            )
            obs = env.reset()
            _ = CompletedRateViewer(
                obs,
                config.TASK.SEMANTIC_TOPDOWN_SENSOR.NUM_TOTAL_CHANNELS,
                object_category_name,
                env,
                env.task.semantic_map_builder,
            )
            plt.show()


@pytest.fixture(scope="class")
def long_term_goal_reachability_measure_config(
    request,
    map_pose_sensor_config,
    navigation_action_config,
    semantic_map_builder_config,
    semantic_topdown_sensor_config,
):
    request.cls.config = map_pose_sensor_config.clone()
    request.cls.config.merge_from_other_cfg(semantic_topdown_sensor_config)
    request.cls.config.merge_from_other_cfg(semantic_map_builder_config)
    request.cls.config.merge_from_other_cfg(navigation_action_config)
    request.cls.config.defrost()
    request.cls.config.TASK.SENSORS = ["MAP_POSE_SENSOR", "SEMANTIC_TOPDOWN_SENSOR"]
    request.cls.config.SIMULATOR.AGENT_0.SENSORS = [
        "DEPTH_SENSOR",
        "SEMANTIC_SENSOR",
        "RGB_SENSOR",
    ]
    request.cls.config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    request.cls.config.TASK.TYPE = "Scanning-v0"
    request.cls.config.TASK.MEASUREMENTS = ["LONG_TERM_GOAL_REACHABILITY"]
    request.cls.config.TASK.LONG_TERM_GOAL_REACHABILITY = Config()
    request.cls.config.TASK.LONG_TERM_GOAL_REACHABILITY.TYPE = (
        "LongTermGoalReachability"
    )
    request.cls.config.freeze()


@pytest.mark.usefixtures("long_term_goal_reachability_measure_config")
class TestLongTermGoalReachabilityMeasure:
    def test_reachable_goal(self):
        with closing(create_env_with_one_episode(self.config)) as env:
            reachable_goal = np.array([-0.175, -0.25])
            action = {
                "action": "NAVIGATION",
                "action_args": {"goal_position": reachable_goal},
            }
            env.reset()
            env.step(action)
            metrics = env.get_metrics()
            assert metrics["long_term_goal_reachability"] < 0.15

    def test_unreachable_goal(self):
        with closing(create_env_with_one_episode(self.config)) as env:
            reachable_goal = np.array([-0.5, -0.25])
            action = {
                "action": "NAVIGATION",
                "action_args": {"goal_position": reachable_goal},
            }
            env.reset()
            env.step(action)
            metrics = env.get_metrics()
            print(metrics)
            assert metrics["long_term_goal_reachability"] > 5.0


@pytest.fixture(scope="class")
def scanning_quality_measure_config(
    request,
    map_pose_sensor_config,
    navigation_action_config,
    semantic_map_builder_config,
    semantic_topdown_sensor_config,
):
    request.cls.config = map_pose_sensor_config.clone()
    request.cls.config.merge_from_other_cfg(semantic_topdown_sensor_config)
    request.cls.config.merge_from_other_cfg(semantic_map_builder_config)
    request.cls.config.merge_from_other_cfg(navigation_action_config)
    request.cls.config.defrost()
    request.cls.config.TASK.SENSORS = ["MAP_POSE_SENSOR", "SEMANTIC_TOPDOWN_SENSOR"]
    request.cls.config.SIMULATOR.AGENT_0.SENSORS = [
        "DEPTH_SENSOR",
        "SEMANTIC_SENSOR",
        "RGB_SENSOR",
    ]
    request.cls.config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    request.cls.config.TASK.TYPE = "Scanning-v0"
    request.cls.config.TASK.MEASUREMENTS = ["SCANNING_QUALITY"]
    request.cls.config.TASK.SCANNING_QUALITY = Config()
    request.cls.config.TASK.SCANNING_QUALITY.TYPE = "ScanningQuality"
    request.cls.config.TASK.SEMANTIC_TOPDOWN_SENSOR.TYPE = "SemanticTopDownCudaSensor"
    request.cls.config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
    request.cls.config.freeze()
    return request.cls.config


@pytest.mark.usefixtures("scanning_quality_measure_config")
class TestScanningQualityMeasure:
    def test_scanning_distance_score(self):
        with closing(create_env_with_one_episode(self.config)) as env:
            env.reset()
            metrics = env.get_metrics()
            assert np.allclose(metrics[ScanningQuality.uuid], 0.2337498)

            env.step({"action": "MOVE_FORWARD", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[ScanningQuality.uuid], 0.2422499)

            env.step({"action": "TURN_LEFT", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[ScanningQuality.uuid], 0.2438749)

            env.step({"action": "TURN_RIGHT", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[ScanningQuality.uuid], 0.2438749)

            env.step({"action": "MOVE_FORWARD", "action_args": None})
            metrics = env.get_metrics()
            assert np.allclose(metrics[ScanningQuality.uuid], 0.2583748)


@pytest.fixture(scope="class")
def scanning_rate_measure_config(
    request, completed_rate_measure_config, scanning_quality_measure_config
):
    request.cls.config = completed_rate_measure_config
    request.cls.config.merge_from_other_cfg(scanning_quality_measure_config)
    request.cls.config.defrost()
    request.cls.config.TASK.MEASUREMENTS = [
        "COMPLETED_AREA",
        "COMPLETED_RATE",
        "SCANNING_QUALITY",
        "SCANNING_RATE",
    ]
    request.cls.config.TASK.SCANNING_RATE = Config()
    request.cls.config.TASK.SCANNING_RATE.TYPE = "ScanningRate"
    request.cls.config.freeze()


@pytest.mark.usefixtures("scanning_rate_measure_config")
class TestScanningRateMeasure:
    def test_scanning_rate_for_finished_scene_with_cuda_observations(self):
        with closing(create_env_with_one_episode(self.config)) as env:
            object_category_name = extract_object_category_name(
                env._sim.semantic_annotations()
            )
            obs = env.reset()
            plt.ioff()
            _ = ScanningRateViewer(
                obs,
                self.config.TASK.SEMANTIC_TOPDOWN_SENSOR.NUM_TOTAL_CHANNELS,
                object_category_name,
                env,
                env.task.semantic_map_builder,
            )
            plt.show()


def test_adaptive_scanning_success(
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
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
    config.DATASET.CONTENT_SCENES = ["5q7pvUzZiYa"]
    config.DATASET.DATA_PATH = "data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz"
    config.freeze()

    env = habitat.Env(config=config, dataset=None)
    valid_start_position = [9.083100318908691, -3.796574831008911, 2.6502721309661865]
    start_rotation = [0, 0.9682338983431397, 0, 0.25004623192371195]

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

    object_category_name = (
        config.TASK.SEMANTIC_TOPDOWN_SENSOR.SEMANTIC_CHANNEL_CATEGORIES
    )
    map_builder = SemanticMapBuilder(config.TASK.SEMANTIC_MAP_BUILDER)
    obs = env.reset()
    # ScanningSuccess.find_belonging_region(env._sim)
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

    env.close()

@pytest.fixture(scope="class")
def object_coverage_measure_config(
    request, completed_rate_measure_config, scanning_quality_measure_config
):
    request.cls.config = completed_rate_measure_config
    request.cls.config.merge_from_other_cfg(scanning_quality_measure_config)
    request.cls.config.defrost()
    request.cls.config.TASK.MEASUREMENTS = [
        "COMPLETED_AREA",
        "COMPLETED_RATE",
        "SCANNING_QUALITY",
        "SCANNING_RATE",
        "OBJECT_COVERAGE"
    ]
    request.cls.config.TASK.SCANNING_RATE = Config()
    request.cls.config.TASK.SCANNING_RATE.TYPE = "ScanningRate"
    request.cls.config.TASK.OBJECT_COVERAGE = Config()
    request.cls.config.TASK.OBJECT_COVERAGE.TYPE = "ObjectCoverage"
    request.cls.config.freeze()

@pytest.mark.usefixtures("object_coverage_measure_config")
class TestObjectCoverageMeasure:
    def test_scanning_rate_for_finished_scene_with_cuda_observations(self):
        # start_position = [-0.0600453,0.0752179,-0.0317102]
        # start_rotation = [0, -0.734314, 0,   0.67881]
        # scene_id = "pLe4wQe7qrG"
        start_position = [8.07306,0.0632534,-0.558928]
        start_rotation = [0,  0.966597,         0, -0.256301]
        scene_id = "2t7WUuJeko7"
        self.config.defrost()
        self.config.DATASET.SPLIT = "val_test"
        self.config.ENVIRONMENT.MAX_EPISODE_STEPS = 1000000
        self.config.freeze()
        with closing(create_mp3d_env_with_one_episode(self.config, scene_id=scene_id, start_position=start_position, start_rotation=start_rotation)) as env:
            object_category_name = extract_object_category_name(
                env._sim.semantic_annotations()
            )
            obs = env.reset()
            plt.ioff()
            _ = ScanningRateViewer(
                obs,
                self.config.TASK.SEMANTIC_TOPDOWN_SENSOR.NUM_TOTAL_CHANNELS,
                object_category_name,
                None,
                env,
                env.task.semantic_map_builder,
            )
            plt.show()
