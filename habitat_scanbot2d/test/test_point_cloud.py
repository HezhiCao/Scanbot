import os
import random

import numpy as np
from habitat.core import spaces

import pytest
import habitat
from habitat.config.default import get_config
from habitat_scanbot2d.point_cloud_sensor import PointCloudSensor
from habitat_scanbot2d.PointCloud import PointCloudProcessing, PointCloudProcess_o3d
import habitat_sim
from habitat_scanbot2d.scanning_task import (
    ScanningEpisode
)
from matplotlib import pyplot as plt
import multiprocessing as mp


def test_glued_point_cloud():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder")
    config.defrost()
    config.TASK.POINT_CLOUD_SENSOR = habitat.Config()
    config.TASK.POINT_CLOUD_SENSOR.TYPE = "PointCloudSensor"
    config.TASK.SENSORS = ["POINT_CLOUD_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.DATASET.CONTENT_SCENES = ["Albertville"]
    config.DATASET.DATA_PATH = (
        "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
    )
    config.freeze()

    with habitat.Env(config=config, dataset=None) as env:
        valid_start_position = [-1.31759, 0.0961462, -3.79719]
        start_rotation = [0., 0., 0., 1.]
        start_rotation = [0, -0.542452, 0, -0.840087]

        env.episode_iterator = iter(
            [
                ScanningEpisode(
                    episode_id="0",
                    scene_id=config.SIMULATOR.SCENE,
                    start_position=valid_start_position,
                    start_rotation=start_rotation
                )
            ]
        )

        max_step = 10
        observations = env.reset()
        count_step = 0
        action_space = spaces.ActionSpace(
            {'MOVE_FORWARD': spaces.EmptySpace(),
             'TURN_LEFT': spaces.EmptySpace(),
             'TURN_RIGHT': spaces.EmptySpace()}
        )
        while count_step < max_step:
            action = action_space.sample()
            print("action:", action['action'])
            observations = env.step(action)
            cur_state = env._sim.get_agent_state()
            if count_step == 0:
                glued_point_cloud = PointCloudProcessing(observations['point_cloud'], cur_state)
            else:
                glued_point_cloud.glue_point_cloud(observations['point_cloud'], cur_state)
                pcd = observations['point_cloud']
            count_step += 1
            if count_step == (max_step - 1):
                process1 = mp.Process(target=glued_point_cloud.view_glued_point_cloud, args=(), daemon=False)
                process1.start()
                glued_point_cloud.pc_to_voxel()
                process2 = mp.Process(target=glued_point_cloud.view_voxel, args=(), daemon=False)
                process2.start()

            # fig = plt.figure(figsize=(25, 15))
            # ax = fig.add_subplot(121)
            # ax.imshow(observations['rgb'])
            # ax = fig.add_subplot(122)
            # ax.imshow(np.squeeze(observations['depth']))
            # plt.show()


def test_glued_point_cloud_o3d():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder")
    config.defrost()
    config.TASK.POINT_CLOUD_SENSOR = habitat.Config()
    config.TASK.POINT_CLOUD_SENSOR.TYPE = "PointCloudSensor"
    config.TASK.SENSORS = ["POINT_CLOUD_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "RGB_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.DATASET.CONTENT_SCENES = ["Albertville"]
    config.DATASET.DATA_PATH = (
        "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
    )
    config.freeze()

    with habitat.Env(config=config, dataset=None) as env:
        valid_start_position = [-1.31759, 0.0961462, -3.79719]
        start_rotation = [0., 0., 0., 1.]
        start_rotation = [0, -0.542452, 0, -0.840087]

        env.episode_iterator = iter(
            [
                ScanningEpisode(
                    episode_id="0",
                    scene_id=config.SIMULATOR.SCENE,
                    start_position=valid_start_position,
                    start_rotation=start_rotation
                )
            ]
        )

        max_step = 10
        observations = env.reset()
        count_step = 0
        action_space = spaces.ActionSpace(
            {'MOVE_FORWARD': spaces.EmptySpace(),
             'TURN_LEFT': spaces.EmptySpace(),
             'TURN_RIGHT': spaces.EmptySpace()}
        )
        while count_step < max_step:
            action = action_space.sample()
            print("action:", action['action'])
            observations = env.step(action)
            cur_state = env._sim.get_agent_state()
            if count_step == 0:
                glued_point_cloud = PointCloudProcess_o3d(observations['rgb'],
                                                          observations['depth'],
                                                          config.SIMULATOR.DEPTH_SENSOR.HFOV, cur_state)
            else:
                glued_point_cloud.glue_point_cloud(observations['rgb'], observations['depth'], cur_state)

            if count_step == (max_step - 1):
                process1 = mp.Process(target=glued_point_cloud.view_glued_point_cloud, args=(), daemon=False)
                process1.start()
                glued_point_cloud.pc_to_voxel()
                process2 = mp.Process(target=glued_point_cloud.view_voxel, args=(), daemon=False)
                process2.start()
            count_step += 1
            # fig = plt.figure(figsize=(25, 15))
            # ax = fig.add_subplot(121)
            # ax.imshow(observations['rgb'])
            # ax = fig.add_subplot(122)
            # ax.imshow(np.squeeze(observations['depth']))
