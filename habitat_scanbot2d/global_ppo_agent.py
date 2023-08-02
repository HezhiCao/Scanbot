#  ____                  _           _
# / ___|  ___ __ _ _ __ | |__   ___ | |_
# \___ \ / __/ _` | '_ \| '_ \ / _ \| __|
#  ___) | (_| (_| | | | | |_) | (_) | |_
# |____/ \___\__,_|_| |_|_.__/ \___/ \__|

"""
    habitat_scanbot2d.global_ppo_agent
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    An agent using trained global policy to autoscanning the scene

    :copyright: (c) 2021 by XiXia.
    :license: MIT License, see LICENSE for more details.
"""

import os
import argparse
import random
from typing import Dict, Any
from itertools import islice
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from gym import spaces

import habitat
from habitat.config import Config
from habitat.core.simulator import Observations
from habitat.core.agent import Agent
from habitat_baselines.config.default import get_config as get_baseline_config
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.obs_transformers import (
    get_active_obs_transforms,
    apply_obs_transforms_obs_space,
    apply_obs_transforms_batch,
)

import habitat_scanbot2d
from habitat_scanbot2d.utils.semantic_map import compute_map_size_in_cells
from habitat_scanbot2d.policies.scanning_global_policy import ScanningGlobalPolicy
from habitat_scanbot2d.utils.semantic_map import extract_object_category_name
from habitat_scanbot2d.utils.visualization import SemanticMapViewer
from habitat_scanbot2d.environments import ScanningRLEnv
from habitat_scanbot2d.measures import (
    CompletedRate,
    CompletedArea,
    ScanningRate,
    ScanningQuality,
    LongTermGoalReachability,
)


class GlobalPPOAgent(Agent):
    def __init__(self, config: Config) -> None:
        task_config = config.TASK_CONFIG
        rl_config = config.RL
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore
            self.device = torch.device(f"cuda:{config.PTH_GPU_ID}")
        else:
            self.device = torch.device("cpu")

        global_map_size_in_cells = compute_map_size_in_cells(
            task_config.TASK.SEMANTIC_TOPDOWN_SENSOR.MAP_SIZE_IN_METERS,
            task_config.TASK.SEMANTIC_TOPDOWN_SENSOR.MAP_CELL_SIZE,
        )
        local_map_size_in_cells = compute_map_size_in_cells(
            task_config.TASK.SEMANTIC_MAP_BUILDER.MAP_SIZE_IN_METERS,
            task_config.TASK.SEMANTIC_MAP_BUILDER.MAP_CELL_SIZE,
        )
        num_channels = (
            task_config.TASK.SEMANTIC_MAP_BUILDER.NUM_TOTAL_CHANNELS * 2
            if task_config.TASK.SEMANTIC_MAP_BUILDER.USE_LOCAL_REPRESENTATION
            else task_config.TASK.SEMANTIC_MAP_BUILDER.NUM_TOTAL_CHANNELS
        )
        self.obs_transformers = get_active_obs_transforms(config)
        observation_spaces = {}
        observation_spaces["global_semantic_map"] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(
                local_map_size_in_cells,
                local_map_size_in_cells,
                num_channels,
            ),
            dtype=np.float32,
        )
        observation_spaces["map_pose"] = spaces.Box(
            low=np.array((0.0, 0.0, -1.0, -1.0), dtype=np.float32),
            high=np.array(
                (
                    global_map_size_in_cells - 1,
                    global_map_size_in_cells - 1,
                    1.0,
                    1.0,
                ),
                dtype=np.float32,
            ),
            shape=(4,),
            dtype=np.float32,
        )

        observation_spaces = apply_obs_transforms_obs_space(
            spaces.Dict(observation_spaces), self.obs_transformers
        )

        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.hidden_size = rl_config.PPO.hidden_size

        self.actor_critic = ScanningGlobalPolicy(
            observation_space=observation_spaces,
            action_space=action_space,
            hidden_size=self.hidden_size,
            rnn_type=rl_config.DDPPO.rnn_type,
            num_recurrent_layers=rl_config.DDPPO.num_recurrent_layers,
            visual_backbone=rl_config.DDPPO.backbone,
            global_ppo_agent=rl_config.PPO.global_backbone,
            normalize_visual_inputs=False,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            policy_config=rl_config.POLICY,
            used_inputs=rl_config.PPO.used_inputs,
        )

        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            # Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )
        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating a random model."
            )

        self.recurrent_hidden_states: torch.Tensor
        self.not_done_masks: torch.Tensor
        self.prev_actions: torch.Tensor

    def reset(self) -> None:
        self.recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.ones(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 2, device=self.device)

    def act(self, observations: Observations) -> Dict[str, Any]:
        batch = batch_obs([observations], device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transformers)  # type: ignore
        with torch.no_grad():
            (_, action, _, self.recurrent_hidden_states) = self.actor_critic.act(
                batch,
                self.recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            self.prev_actions.copy_(action)

        goal_position = torch.clip(action, min=-1.0, max=1.0).squeeze()
        goal_mean = torch.clip(self.actor_critic.mean, min=-1.0, max=1.0).squeeze()

        return {
            "action": "NAVIGATION",
            "action_args": {
                "goal_position": goal_position.cpu().numpy(),
                "goal_mean": goal_mean.cpu().numpy(),
                "goal_stddev": self.actor_critic.stddev.squeeze().cpu().numpy(),
            },
        }


def get_config(config_path: str, model_path: str):
    if os.path.exists(config_path):
        config = get_baseline_config(config_path)
    else:
        habitat.logger.warning("Invalid config path!")
        config = get_baseline_config()
    config.defrost()
    config.RANDOM_SEED = 106
    config.PTH_GPU_ID = 0
    config.TASK_CONFIG.DATASET.SPLIT = "val_test"
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = [
        # "17DRP5sb8fy",
        # "1LXtFkjw3qL",
        # "1pXnuDYAj8r",
        # "29hnd4uzFmX",
        # "2n8kARJN3HM",
        # "5LpN3gDmAk7",
        # "5q7pvUzZiYa",
        # "759xd9YjKW5",
        # "7y3sRwLe3Va",
        # "e9zR4mvMWw7",
        # "JeFG25nYj2p",
        # "i5noydFURQK",
        # "b8cTxDM8gDG",
        # "cV4RVeZvu5T",
        # "D7G3Y4RVNrH",
        # "D7N2EKCX4Sj",
        # "pRbA3pwrgk9",
        # "VVfe2KiqLaN",
        # "JF19kD82Mey",
        # "HxpKQynjfin", # a small scene, with some unreachable region
        # "pRbA3pwrgk9", # roof episode, with stairs
        # "GdvgFV5R1Z5", # a small scene
        # "sT4fr6TAbpF",
        # "uNb9QFRL6hY", # has to many stairs
        # "XcA2TqTSSAj",
        # "Albertville",
        # "Goffs",
        # "Arkansaw",
        # "Andover",
        # "Hillsdale",
        # "Hainesburg",
        # "Shelbiana",
        # "Oyens",
        # "2azQ1b91cZZ", # validate set begin
        # "8194nk5LbLH",
        # "EU6Fwq7SyZv",
        # "QUCTc6BB5sX",
        "TbHJrupSAjP",
        # "X7HyMhZNoso",
        # "Z6MFQCViBuw",
        # "oLBMNvg9in8",
        # "pLe4wQe7qrG",
        # "x8F5xyUWy9e",
        # "zsNo4HB9uLZ",
        # "2t7WUuJeko7", # test set begin
        # "5ZKStnWn8Zo",
        # "ARNzJeq3xxb",
        # "RPmz2sHmrrY",
        # "UwV83HsGsw3",
        # "Vt2qJdWjCF2",
        # "WYY7iVyf5p8",
        # "YFuZgdQ5vWj",
        # "YVUC4YcDtcY",
        # "fzynW3qQPVF",
        # "gYvKGZ5eRqb", # church
        # "gxdoqLR6rwA",
        # "jtcxE69GiFV",
        # "pa4otMbVnkk",
        # "q9vSo1VnCiC",
        # "rqfALeAoiTq",
        # "wc2JMjhGNzB",
        # "yqstnuAEVhm",
    ]
    config.TASK_CONFIG.TASK.ACTIONS.NAVIGATION.VISUALIZATION = True
    config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 200
    config.SENSORS = ["DEPTH_SENSOR", "SEMANTIC_SENSOR", "RGB_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = [
        "DEPTH_SENSOR",
        "SEMANTIC_SENSOR",
        "RGB_SENSOR",
    ]
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = 480
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = 640
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = (
        1000.0  # don't clamp max depth
    )
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 480
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 640
    config.TASK_CONFIG.TASK.POINT_CLOUD_RECONSTRUCTOR = Config()
    config.TASK_CONFIG.TASK.POINT_CLOUD_RECONSTRUCTOR.TYPE = (
        "PointCloudReconstructorSensor"
    )
    config.TASK_CONFIG.TASK.POINT_CLOUD_RECONSTRUCTOR.H_MAX = 0.8
    config.TASK_CONFIG.TASK.POINT_CLOUD_RECONSTRUCTOR.F_MAX = 3.0
    config.TASK_CONFIG.TASK.POINT_CLOUD_RECONSTRUCTOR.VOXEL_SIZE = 0.01
    config.TASK_CONFIG.TASK.POINT_CLOUD_RECONSTRUCTOR.POINT_CLOUD_SENSOR = Config()
    config.TASK_CONFIG.TASK.POINT_CLOUD_RECONSTRUCTOR.POINT_CLOUD_SENSOR.TYPE = (
        "PointCloudCudaSensor"
    )
    # config.TASK_CONFIG.TASK.SENSORS.append("POINT_CLOUD_RECONSTRUCTOR")
    # config.TASK_CONFIG.TASK.SENSORS.append("CATEGORY_SEMANTIC_SENSOR")
    config.TASK_CONFIG.TASK.SCANNING_SUCCESS.USE_ADAPTIVE_RATE = False
    config.TASK_CONFIG.TASK.SCANNING_SUCCESS.SUCCESS_COMPLETED_RATE = 0.8
    config.TASK_CONFIG.TASK.SCANNING_SUCCESS.SUCCESS_SCANNING_RATE = 0.6

    if os.path.exists(model_path):
        config.MODEL_PATH = model_path
    else:
        habitat.logger.warning("Invalid config path!")
    config.freeze()
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config-path",
        help="Baseline config file path",
        type=str,
        default="configs/scanning_rl_mp3d.yaml",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        help="Checkpoint model path (.pth)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--skip-episode",
        help="Skip several episodes from the dataset",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-e",
        "--episode-num",
        help="How many episodes to test",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-b",
        "--blacklist",
        help="Add current episode to the blacklist file",
        type=str,
        nargs="?",
        const="episode_backlist.txt",
    )

    args = parser.parse_args()
    config = get_config(args.config_path, args.model_path)
    agent = GlobalPPOAgent(config)
    agent.reset()

    env = ScanningRLEnv(config=config)
    env._env.episode_iterator = islice(
        env._env.episode_iterator, args.skip_episode, None
    )
    # env._env._task.replanning_times = 0
    obs = env.reset()
    print(env.current_episode)
    if args.blacklist is not None:
        with open(args.blacklist, "a") as blacklist_file:
            scene_name = Path(env.current_episode.scene_id).stem
            blacklist_file.write(f"{scene_name},{env.current_episode.episode_id}\n")
        return

    episode_num = args.episode_num
    scene_recon_stats = [[] for _ in range(200)]
    roi_count = 0
    step_count = 0
    # replanning_times = []
    long_term_goals = []
    while episode_num > 0:
        action = agent.act(obs)
        obs, *_ = env.step(action=action)
        step_count += env._env.task.primitive_actions_in_last_navigation
        long_term_goals.append(obs["previous_goal_position"] * 360 + 360)
        print(f"Completed rate: {env._env.get_metrics()[CompletedRate.uuid]}")
        # print(f"Completed rate: {env._env.get_metrics()[CompletedArea.uuid]}")
        # print(f"Scanning rate: {env._env.get_metrics()[ScanningRate.uuid]}")
        # print(f"Scanning quality: {env._env.get_metrics()[ScanningQuality.uuid]}")
        # print(
        #     f"Long term goal reachability: {env._env.get_metrics()[LongTermGoalReachability.uuid]}"
        # )
        scene_reconstruction_rate = (
            env._env.get_metrics()[CompletedRate.uuid] * 0.4
            + env._env.get_metrics()[ScanningRate.uuid] * 0.6
        )
        print(f"Scene Reconstruction: {scene_reconstruction_rate}")
        scene_recon_stats[roi_count].append(scene_reconstruction_rate)
        roi_count += 1
        if env.get_done(obs):
        # if roi_count >= 20 or env._env.get_metrics()[CompletedRate.uuid] > 0.85:
            print(f"roi: {roi_count}")
            print(f"step: {step_count}")
            # plt.imshow(obs["global_semantic_map"][..., 0].cpu().numpy(), vmin=0.0, vmax=1.0)
            # for i, goal in enumerate(long_term_goals):
            #     plt.text(goal[1], goal[0], str(i), color="red")
            # np.save("plots/chinagraph/visual_map.npy", obs["global_semantic_map"].cpu().numpy())
            # pose_trajectory = env._env.task.sensor_suite.sensors["map_pose"].pose_trajectory
            # np.save("plots/chinagraph/pose_trajectory.npy", np.array(pose_trajectory))
            # replanning_times.append(env._env._task.replanning_times)
            # env._env._task.replanning_times = 0
            obs = env.reset()
            roi_count = 0
            episode_num -= 1
            print(env.current_episode)

            # final_pcd = o3d.geometry.PointCloud()
            # whole_point_cloud = obs["point_cloud_reconstructor"].cpu().numpy()
            # point_xyz = np.empty((whole_point_cloud.shape[0], 3))
            # point_xyz[:, 0] = whole_point_cloud[:, 0]
            # point_xyz[:, 1] = -whole_point_cloud[:, 2]
            # point_xyz[:, 2] = whole_point_cloud[:, 1]
            # final_pcd.points = o3d.utility.Vector3dVector(point_xyz)
            # final_pcd.colors = o3d.utility.Vector3dVector(
            #     (whole_point_cloud[:, 3:6] / 255.0)
            # )
            # o3d.visualization.draw_geometries([final_pcd])
            # o3d.io.write_point_cloud("agent_point_cloud.ply", final_pcd)
            print(env._env.task.sensor_suite.sensors.keys())
            # np.save(
            #     "agent_category_mapping.npy",
            #     env._env.task.sensor_suite.sensors["category_semantic_cuda"]
            #     .category_mapping.cpu()
            #     .numpy(),
            # )
            # plt.ioff()
            # plt.show()
            # break

    final_stats = []
    for step_stats in scene_recon_stats:
        final_stats.append(
            [
                np.mean(step_stats),
                np.var(step_stats),
                np.min(step_stats),
                np.max(step_stats),
            ]
        )
        print(f"Mean: {np.mean(step_stats)}, Var: {np.var(step_stats)}")

    # np.save("plots/scene_reconstruction_timeline.npy", np.array(final_stats))
    # print(f"replanning times: {np.mean(replanning_times)}")


if __name__ == "__main__":
    main()
