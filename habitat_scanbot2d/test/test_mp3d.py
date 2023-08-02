import os
from itertools import cycle
from contextlib import closing
import pytest

import habitat
from habitat.config.default import get_config

from habitat_scanbot2d.scanning_task import ScanningEpisode
from habitat_scanbot2d.utils.visualization import visualize_semantic


def create_mp3d_env_with_one_episode(
    config, scene_id=None, split=None, start_position=None, start_rotation=None
) -> habitat.Env:
    config.defrost()
    if scene_id is not None:
        config.DATASET.CONTENT_SCENES = [scene_id]
    else:
        config.DATASET.CONTENT_SCENES = ["zsNo4HB9uLZ"]
        # config.DATASET.CONTENT_SCENES = ["1LXtFkjw3qL"]
    if split is not None:
        config.DATASET.SPLIT = split
    else:
        config.DATASET.SPLIT = "train"
    config.DATASET.DATA_PATH = "data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz"
    config.freeze()
    env = habitat.Env(config=config, dataset=None)
    if start_position is not None:
        valid_start_position = start_position
    else:
        valid_start_position = [0.820116,0.0716274,-6.80947]
        # valid_start_position = [-0.959103, -2.91559, 10.326]

    if start_rotation is not None:
        valid_start_rotation = start_rotation
    else:
        valid_start_rotation = [0, -0.99307, 0, -0.117525]
        # valid_start_rotation = [0, -0.0549381, 0, 0.99849]

    env.episode_iterator = cycle(
        [
            ScanningEpisode(
                episode_id="0",
                scene_id=config.SIMULATOR.SCENE,
                start_position=valid_start_position,
                start_rotation=valid_start_rotation,
            )
        ]
    )
    return env


def create_mp3d_env_with_unseen_obstacles(config) -> habitat.Env:
    config.defrost()
    config.DATASET.CONTENT_SCENES = ["VVfe2KiqLaN"]
    config.DATASET.DATA_PATH = "data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz"
    config.freeze()

    env = habitat.Env(config=config, dataset=None)
    valid_start_position = [0.346170, 0.152306, 0.498264]
    start_rotation = [0, 0.945983, 0, 0.324215]

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

    with closing(create_mp3d_env_with_one_episode(config)) as env:
        init_obs = env.reset()
        init_semantic = init_obs["category_semantic"]
        visualize_semantic(init_semantic)
