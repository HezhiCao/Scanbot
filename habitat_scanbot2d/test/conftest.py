import os
import pytest

import numpy as np
from habitat.config.default import Config, get_config
from habitat_scanbot2d.scanning_task import ScanningEpisode


def check_data_folder(config: Config):
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")


@pytest.fixture(scope="session")
def map_pose_sensor_config():
    config = get_config()
    check_data_folder(config)
    config.defrost()
    config.TASK.MAP_POSE_SENSOR = Config()
    config.TASK.MAP_POSE_SENSOR.TYPE = "MapPoseSensor"
    config.TASK.MAP_POSE_SENSOR.MAP_SIZE_IN_METERS = 36
    config.TASK.MAP_POSE_SENSOR.MAP_CELL_SIZE = 0.05
    config.freeze()
    return config


@pytest.fixture(scope="session")
def semantic_topdown_sensor_config():
    config = get_config()
    check_data_folder(config)
    config.defrost()
    config.TASK.SEMANTIC_TOPDOWN_SENSOR = Config()
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.TYPE = "SemanticTopDownSensor"
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.NEAR_THRESHOLD = 0.1
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.FAR_THRESHOLD = 4.0
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.H_MIN_THRESHOLD = 0.1
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.H_MAX_THRESHOLD = 2.0
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.INITIAL_CAMERA_HEIGHT = (
        config.SIMULATOR.DEPTH_SENSOR.POSITION[1]
    )
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.MAP_SIZE_IN_METERS = 36
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.MAP_CELL_SIZE = 0.05
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.NUM_QUALITY_CHANNELS = 8
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.BEST_SCANNING_DISTANCE = 1.5
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.NUM_TOTAL_CHANNELS = 21
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.DATASET_TYPE = "mp3d"
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.SEMANTIC_CHANNEL_CATEGORIES = [
        "chair",
        "table",
        "cabinet",
        "sofa",
        "bed",
        "sink",
        "stairs",
        "bathtub",
        "counter",
        "others",
    ]
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.MIN_POINT_NUM_THRESHOLD = 20
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.USE_TORCH_EXTENSIONS = True
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.POINT_CLOUD_SENSOR = Config()
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.POINT_CLOUD_SENSOR.TYPE = "PointCloudSensor"
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.CATEGORY_SEMANTIC_SENSOR = Config()
    config.TASK.SEMANTIC_TOPDOWN_SENSOR.CATEGORY_SEMANTIC_SENSOR.TYPE = (
        "CategorySemanticSensor"
    )
    config.freeze()
    return config


@pytest.fixture
def point_cloud_sensor_config():
    config = get_config()
    check_data_folder(config)
    config.defrost()
    config.TASK.POINT_CLOUD_SENSOR = Config()
    config.TASK.POINT_CLOUD_SENSOR.TYPE = "PointCloudSensor"
    config.freeze()
    return config


@pytest.fixture(scope="session")
def category_semantic_sensor_config():
    config = get_config()
    check_data_folder(config)
    config.defrost()
    config.TASK.CATEGORY_SEMANTIC_SENSOR = Config()
    config.TASK.CATEGORY_SEMANTIC_SENSOR.TYPE = "CategorySemanticSensor"
    config.freeze()
    return config


@pytest.fixture(scope="session")
def primitive_action_planner_config():
    config = get_config()
    check_data_folder(config)
    config.defrost()
    config.TASK.PRIMITIVE_ACTION_PLANNER = Config()
    config.TASK.PRIMITIVE_ACTION_PLANNER.MAP_SIZE_IN_METERS = 36
    config.TASK.PRIMITIVE_ACTION_PLANNER.MAP_CELL_SIZE = 0.05
    config.TASK.PRIMITIVE_ACTION_PLANNER.ANGLE_THRESHOLD = float(np.deg2rad(15))
    config.TASK.PRIMITIVE_ACTION_PLANNER.POSITION_THRESHOLD = 0.15
    config.TASK.PRIMITIVE_ACTION_PLANNER.NEXT_WAYPOINT_THRESHOLD = 0.5
    config.freeze()
    return config


@pytest.fixture(scope="session")
def navigation_action_config(primitive_action_planner_config):
    config = get_config()
    check_data_folder(config)
    config.defrost()
    config.TASK.ACTIONS.NAVIGATION = Config()
    navigation_config = config.TASK.ACTIONS.NAVIGATION
    navigation_config.TYPE = "NavigationAction"
    navigation_config.VISUALIZATION = True
    navigation_config.BASELINE = Config()
    navigation_config.BASELINE.PLANNER_MAX_STEPS = 500
    navigation_config.BASELINE.PREPROCESS_MAP = True
    navigation_config.BASELINE.BETA = 100
    navigation_config.MAX_PRIMITIVE_STEPS = 1000
    navigation_config.COLLIDED_PLANNING_THRESHOLD = 10
    navigation_config.PATH_PLANNING_INTERVAL = 10
    navigation_config.PATH_FINDER_OBSTACLE_COST = 10000.0
    navigation_config.PATH_FINDER_ITERATION_THRESHOLD = -1
    navigation_config.USE_SIMULATOR_ORACLE = False
    navigation_config.OBSTACLE_THRESHOLD = 10
    navigation_config.NUM_TURN_AROUND = 0
    navigation_config.COUNTED_STEP_THRESHOLD = 2
    navigation_config.PRIMITIVE_ACTION_PLANNER = (
        primitive_action_planner_config.TASK.PRIMITIVE_ACTION_PLANNER
    )
    config.TASK.POSSIBLE_ACTIONS = [
        "STOP",
        "MOVE_FORWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "NAVIGATION",
    ]
    config.freeze()
    return config


@pytest.fixture(scope="session")
def semantic_map_builder_config():
    config = get_config()
    check_data_folder(config)
    config.defrost()
    config.TASK.SEMANTIC_MAP_BUILDER = Config()
    config.TASK.SEMANTIC_MAP_BUILDER.MAP_SIZE_IN_METERS = 36
    config.TASK.SEMANTIC_MAP_BUILDER.MAP_CELL_SIZE = 0.15
    config.TASK.SEMANTIC_MAP_BUILDER.NUM_TOTAL_CHANNELS = 21
    config.TASK.SEMANTIC_MAP_BUILDER.USE_LOCAL_REPRESENTATION = True
    config.TASK.SEMANTIC_MAP_BUILDER.LOCAL_MAP_CELL_SIZE = 0.05
    config.TASK.SEMANTIC_MAP_BUILDER.TRAJECTORY_LENGTH = 100
    config.freeze()
    return config
