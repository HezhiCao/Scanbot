#  ____                  _           _
# / ___|  ___ __ _ _ __ | |__   ___ | |_
# \___ \ / __/ _` | '_ \| '_ \ / _ \| __|
#  ___) | (_| (_| | | | | |_) | (_) | |_
# |____/ \___\__,_|_| |_|_.__/ \___/ \__|

"""
    habitat_scanbot2d.global_ppo_trainer
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    A class to train global exploration policy using Proximal Policy Optimization (PPO)

    :copyright: (c) 2021 by XiXia.
    :license: MIT License, see LICENSE for more details.
"""

import contextlib
import time
import random
import copy
from collections import defaultdict, deque
from itertools import chain
from pathlib import Path
from typing import Dict, Any, cast

import numpy as np
import torch
import tqdm
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from gym import spaces

from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_scanbot2d.auxiliary_tasks.auxiliary_rollout_storage import (
    AuxiliaryRoulloutStorage,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_scanbot2d.auxiliary_tasks.auxiliary_ppo import AuxiliaryPPO, AuxiliaryDDPPO
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    save_resume_state,
    rank0_only,
    requeue_job,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_scanbot2d.measures import (
    LeftStepCount,
    RightStepCount,
    QualityIncreaseRatio,
)
from habitat_scanbot2d.auxiliary_tasks import get_auxiliary_task
from habitat_scanbot2d.utils.visualization import observations_to_image


@baseline_registry.register_trainer(name="ppo-scanning")
class ScanningPPOTrainer(PPOTrainer):
    supported_tasks = ["Scanning-v0"]
    METRICS_BLACKLIST = {
        "scene_id",
        "left_step_count",
        "right_step_count",
    }

    def __init__(self, config):
        super().__init__(config=config)
        self.using_navigation_action = (
            "NAVIGATION" in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        )
        self.auxiliary_tasks = []

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats["_last_checkpoint_percent"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(requeue_stats["window_episode_stats"])

        ppo_cfg = self.config.RL.PPO

        with (
            TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs)
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self.agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(ppo_cfg.num_steps):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == ppo_cfg.num_steps
                    )

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                profiling_wrapper.range_push("_collect_rollout_step")

                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                    aux_task_loss,
                ) = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        value_loss=value_loss,
                        action_loss=action_loss,
                        dist_entropy=dist_entropy,
                        aux_task_loss=aux_task_loss,
                    ),
                    count_steps_delta,
                )

                if self.num_updates_done % self.config.LOG_ACHIEVED_RATE_INTERVAL == 0:
                    self._all_gather_achieved_rate_interval()

                self._training_log(writer, losses, prev_time)

                if (
                    self.config.USE_LINEAR_REACHABILITY_SCHEDULER
                    and self.num_updates_done
                    % self.config.LINEAR_REACHABILITY_SCHEDULER_UPDATE_INTERVAL
                    == 0
                ):
                    current_reachability_reward = self.envs.call(
                        ["set_reachability_reward"] * self.envs.num_envs,
                        [
                            {
                                "percentage": min(
                                    self.num_steps_done
                                    / self.config.LINEAR_REACHABILITY_SCHEDULER_MAX_STEP,
                                    1.0,
                                )
                            }
                        ]
                        * self.envs.num_envs,
                    )
                    logger.info(
                        f"current reachability reward: {current_reachability_reward[0]}"
                    )

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _init_train(self):
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config: Config = resume_state["config"]
            self.using_navigation_action = (
                "NAVIGATION" in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
            )

        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            self._init_distributed_settings()
        else:
            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        if self.using_navigation_action:
            self.policy_action_space = self.envs.action_spaces[0]["NAVIGATION"]
            action_shape = self.policy_action_space.shape
            discrete_actions = False
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = None
            discrete_actions = True

        if self.config.USE_LINEAR_REACHABILITY_SCHEDULER:
            self.envs.call(
                ["set_reachability_reward"] * self.envs.num_envs,
                [{"percentage": 0.0}] * self.envs.num_envs,
            )

        ppo_config = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only():
            Path(self.config.CHECKPOINT_FOLDER).mkdir(exist_ok=True)

        # initialize auxiliary tasks
        self._setup_auxiliary_tasks(
            ppo_config,
            self.config.RL.AUXILIARY_TASKS,
            num_recurrent_layers=self.config.RL.DDPPO.num_recurrent_layers,
        )

        self._setup_actor_critic_agent(ppo_config)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.semantic_map_encoder
            obs_space = spaces.Dict(
                {
                    "semantic_map_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if ppo_config.use_double_buffered_sampler else 1

        self.rollouts = AuxiliaryRoulloutStorage(
            ppo_config.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            ppo_config.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_config.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._static_encoder:
            with torch.no_grad():
                batch["semantic_map_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

        self._init_statistics(ppo_config.reward_window_size)

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[self.rollouts.current_rollout_step_idx]

            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        self.agent.train()

        value_loss, action_loss, dist_entropy, aux_task_loss = self.agent.update(
            self.rollouts
        )

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (value_loss, action_loss, dist_entropy, aux_task_loss)

    def _init_distributed_settings(self):
        r"""Initialize distributed system and set rank related configs"""
        local_rank, tcp_store = init_distrib_slurm(self.config.RL.DDPPO.distrib_backend)
        if rank0_only():
            logger.info(
                "Initialized DD-PPO with {} workers".format(
                    torch.distributed.get_world_size()
                )
            )

        self.config.defrost()
        self.config.TORCH_GPU_ID = local_rank
        self.config.SIMULATOR_GPU_ID = local_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += (
            torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
        )
        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)
        self.num_rollouts_done_store = torch.distributed.PrefixStore(
            "rollout_tracker", tcp_store
        )
        self.num_rollouts_done_store.set("num_done", "0")

    def _init_statistics(self, reward_window_size):
        r"""Initialize statistic values to keep track of episode information for each environments
        or scenes
        running_episode_stats:
            * reward: accumulated reward for finished episodes
            * count: finished episodes
            * other scalars: infos returned by RLEnv.step()
        window_episode_stats:
            same statistics for all distributed workers, but only keep recent window size items
        _scene_statistics:
            running metric statistics for all appeared scenes
        _window_scene_statistics:
            same statistics for all workers, but only keep a limited window size

        :param reward_window_size: window size to for computing
        """
        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = {
            "count": torch.zeros(self.envs.num_envs, 1),
            "reward": torch.zeros(self.envs.num_envs, 1),
        }
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=reward_window_size)
        )
        self._scene_statistics = {}
        self._window_scene_statistics = defaultdict(
            lambda: deque(maxlen=reward_window_size)
        )

    def _setup_auxiliary_tasks(self, ppo_cfg, aux_cfg, **kwargs):
        self.auxiliary_tasks.clear()
        for task in aux_cfg.tasks:
            task_class = get_auxiliary_task(task)
            auxiliary_task = task_class(ppo_cfg, aux_cfg, self.device, **kwargs)
            self.auxiliary_tasks.append(auxiliary_task)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        :param ppo_cfg: config node with relevant params
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config,
            observation_space,
            self.policy_action_space,
            auxiliary_tasks=self.auxiliary_tasks,
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if self.config.RL.DDPPO.pretrained_encoder or self.config.RL.DDPPO.pretrained:
            self._load_pretrained_weights()

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = (AuxiliaryDDPPO if self._is_distributed else AuxiliaryPPO)(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            auxiliary_tasks=self.auxiliary_tasks,
            auxiliary_loss_coef=self.config.RL.AUXILIARY_TASKS.loss_coef,
        )

        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)  # type: ignore

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                action_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for env_index, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if self.using_navigation_action:
                step_action = self._convert_actor_output_to_navigation_action(act)
            else:
                step_action = act.item()
            self.envs.async_step_at(env_index, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )

    def _convert_actor_output_to_navigation_action(
        self, actor_output: torch.Tensor
    ) -> Dict[str, Any]:
        r"""Construct a navigation action from the output of the actor

        :param actor_output: (2,) float tensor
        :return: Action dict with "action" and "action_args"
        """
        goal_position = torch.clip(actor_output, min=-1.0, max=1.0)
        return {
            "action": {
                "action": "NAVIGATION",
                "action_args": {"goal_position": goal_position.numpy()},
            }
        }

    def _load_pretrained_weights(self):
        r"""Load either all pretrained weights for actor critic or visual encoder only"""
        pretrained_state = torch.load(
            self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
        )

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        else:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        aux_cfg = self.config.RL.AUXILIARY_TASKS
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l, dtype=torch.float, device=self.current_episode_reward.device
        ).unsqueeze(1)
        try:
            left_step_count = torch.tensor(
                [info[LeftStepCount.uuid] for info in infos],
                dtype=torch.int32,
                device=self.current_episode_reward.device,
            )
            left_step_count = (
                    torch.div(
                        torch.clip(
                            left_step_count,
                            min=0,
                            max=aux_cfg.step_count_clip,
                        ),
                        aux_cfg.step_count_class_interval,
                        rounding_mode="floor",
                    )
                ).unsqueeze(1)
        except KeyError:
            left_step_count = None
        try:
            right_step_count = torch.tensor(
                [info[RightStepCount.uuid] for info in infos],
                dtype=torch.int32,
                device=self.current_episode_reward.device,
            )
            right_step_count = (
                    torch.div(
                        torch.clip(
                            right_step_count,
                            min=0,
                            max=aux_cfg.step_count_clip,
                        ),
                        aux_cfg.step_count_class_interval,
                        rounding_mode="floor",
                    )
                ).unsqueeze(1) # default config: three class: 0-1, 2-3, greater than 3
        except KeyError:
            right_step_count = None
        try:
            quality_increase_ratio = torch.tensor(
                [info[QualityIncreaseRatio.uuid] for info in infos],
                dtype=torch.float32,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
        except KeyError:
            quality_increase_ratio = None
        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(
            done_masks, current_ep_reward.new_zeros(())
        )
        self.running_episode_stats["count"][env_slice] += done_masks.float()
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(
                done_masks, v.new_zeros(())
            )

        if self.config.RECORD_SCENE_STATISTICS:
            self._update_scene_statistics(
                infos, dones, self.current_episode_reward[env_slice]
            )

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)
        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
            left_step_count=left_step_count,
            right_step_count=right_step_count,
            quality_increase_ratio=quality_increase_ratio,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    def _update_scene_statistics(self, infos, dones, rewards):
        if len(infos) == 0:
            return
        stats_ordering = sorted(infos[0].keys())
        for info, done, reward in zip(infos, dones, rewards):
            if not done:
                continue
            # scene_id is a relative path to the glb file
            scene_name = Path(info["scene_id"]).stem
            if scene_name not in self._scene_statistics:
                self._scene_statistics[scene_name] = {
                    k: info[k] for k in stats_ordering if k != "scene_id"
                }
                self._scene_statistics[scene_name]["count"] = 1
                self._scene_statistics[scene_name]["reward"] = reward.item()
            else:
                for k, v in info.items():
                    if k != "scene_id":
                        self._scene_statistics[scene_name][k] += v
                self._scene_statistics[scene_name]["count"] += 1
                self._scene_statistics[scene_name]["reward"] += reward.item()

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        self._all_gather_scene_statistics()

        return super()._coalesce_post_step(losses, count_steps_delta)

    @rank0_only
    def _training_log(self, writer, losses: Dict[str, float], prev_time: int = 0):
        super()._training_log(writer, losses, prev_time=prev_time)

        scene_stat_deltas = {
            scene_name: {
                stat_name: stats[-1][stat_name] - stats[0][stat_name]
                if len(stats) > 1
                else stats[0][stat_name]
                for stat_name in stats[0]
            }
            for scene_name, stats in self._window_scene_statistics.items()
        }

        scene_rewards = {
            scene_name: scene_stat_deltas[scene_name]["reward"]
            / scene_stat_deltas[scene_name]["count"]
            for scene_name in self.config.MONITORED_SCENES
            if scene_name in scene_stat_deltas
        }
        if len(scene_rewards) > 0:
            writer.add_scalars(
                "scene_rewards",
                scene_rewards,
                self.num_steps_done,
            )

        if self.num_updates_done % self.config.LOG_ACHIEVED_RATE_INTERVAL == 0:
            if len(self._achieved_rate_statistics) > 0:
                writer.add_scalars(
                    "scene_rates",
                    self._achieved_rate_statistics,
                    self.num_steps_done,
                )

        if (
            self.config.RECORD_SCENE_STATISTICS
            and self.num_updates_done % self.config.LOG_SCENE_STATISTICS_INTERVAL == 0
            and len(scene_stat_deltas) > 0
        ):
            sorted_scene_stats = sorted(
                scene_stat_deltas.items(),
                key=lambda k_v: -k_v[1]["reward"] / k_v[1]["count"],
            )
            logger.info(
                "\n======= scene statistics =======\n"
                "{}\n".format(
                    "\n".join(
                        "- {}:\t{}".format(
                            scene_name,
                            "  ".join(
                                "{}: {:.3f}".format(k, v / stats["count"])
                                for k, v in stats.items()
                                if k != "count"
                            ),
                        )
                        for scene_name, stats in sorted_scene_stats
                    )
                )
            )

    def _all_gather_scene_statistics(self):
        if self._is_distributed:
            gathered_stats = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered_stats, self._scene_statistics)
        else:
            gathered_stats = [copy.deepcopy(self._scene_statistics)]

        if rank0_only():
            all_scene_statistics = {}
            for stats in gathered_stats:
                for scene_name, stat in stats.items():  # type: ignore
                    if scene_name not in all_scene_statistics:
                        all_scene_statistics[scene_name] = stat
                    else:
                        for metric_name, metric_value in stat.items():
                            all_scene_statistics[scene_name][
                                metric_name
                            ] += metric_value

            for scene_name in all_scene_statistics:
                try:
                    if (
                        all_scene_statistics[scene_name]["count"]
                        > self._window_scene_statistics[scene_name][-1]["count"]
                    ):
                        self._window_scene_statistics[scene_name].append(
                            all_scene_statistics[scene_name]
                        )
                except IndexError:
                    self._window_scene_statistics[scene_name].append(
                        all_scene_statistics[scene_name]
                    )

    def _all_gather_achieved_rate_interval(self):
        stats = self.envs.call(["get_achieved_rate_statistics"] * self.envs.num_envs)
        if self._is_distributed:
            gathered_stats = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered_stats, stats)
        else:
            gathered_stats = [stats]

        gathered_stats = chain.from_iterable(gathered_stats)  # type: ignore
        stat_names = [
            "max_completed_rate",
            "max_scanning_rate",
            "min_completed_rate",
            "min_scanning_rate",
            "average_completed_rate",
            "average_scanning_rate",
        ]
        reduction_fns = [max, max, min, min, np.mean, np.mean]

        if rank0_only():
            self._achieved_rate_statistics = {}
            for stats in gathered_stats:
                for scene_name, stat in stats.items():  # type: ignore
                    if scene_name not in self.config.MONITORED_SCENES:
                        continue

                    if (
                        f"{scene_name}_max_completed_rate"
                        not in self._achieved_rate_statistics
                    ):
                        for stat_name in stat_names:
                            self._achieved_rate_statistics[
                                f"{scene_name}_{stat_name}"
                            ] = stat[f"{stat_name}"]
                    else:
                        for stat_name, reduction_fn in zip(stat_names, reduction_fns):
                            self._achieved_rate_statistics[
                                f"{scene_name}_{stat_name}"
                            ] = reduction_fn(
                                (
                                    self._achieved_rate_statistics[
                                        f"{scene_name}_{stat_name}"
                                    ],
                                    stat[f"{stat_name}"],
                                )
                            )

    def _eval_checkpoint(
        self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0
    ) -> None:
        r"""Evaluates a single checkpoint.

        :param checkpoint_path: path of checkpoint
        :param writer: tensorboard writer object for logging to tensorboard
        :param checkpoint_index: index of current checkpoint for logging
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_config = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

        if self.using_navigation_action:
            self.policy_action_space = self.envs.action_spaces[0]["NAVIGATION"]
            action_shape = self.policy_action_space.shape
            action_type = torch.float
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = (1,)
            action_type = torch.long

        self._setup_actor_critic_agent(ppo_config)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            self.envs.num_envs,
            self.actor_critic.net.num_recurrent_layers,
            ppo_config.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.envs.num_envs,
            *action_shape,
            device=self.device,
            dtype=action_type,
        )
        not_done_masks = torch.zeros(
            self.envs.num_envs,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes = {}

        rgb_frames = [[] for _ in range(self.config.NUM_ENVIRONMENTS)]
        if len(self.config.VIDEO_OPTION) > 0:
            Path(self.config.VIDEO_DIR).mkdir(exist_ok=True)

        num_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        total_num_episodes = cast(int, sum(self.envs.number_of_episodes))
        if num_of_eval_episodes == -1:
            num_of_eval_episodes = total_num_episodes
        else:
            if total_num_episodes < num_of_eval_episodes:
                logger.warn(
                    f"Config specified {num_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_episodes}."
                )
                logger.warn(f"Evaluating with {total_num_episodes} instead.")
                num_of_eval_episodes = total_num_episodes

        pbar = tqdm.tqdm(total=num_of_eval_episodes)
        self.actor_critic.eval()
        # paused environment will decrease VectorEnv.num_envs
        while len(stats_episodes) < num_of_eval_episodes and self.envs.num_envs > 0:
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            if self.using_navigation_action:
                step_actions = [
                    self._convert_actor_output_to_navigation_action(act)
                    for act in actions.to(device="cpu")
                ]
            else:
                step_actions = [act.item() for act in actions.to(device="cpu")]

            outputs = self.envs.step(step_actions)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            for i in range(self.envs.num_envs):
                # Recurring episode_id indicates that this environment is running out of episodes
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(self._extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    # use (scene_id, episode_id) as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )
                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        max_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
            )
            max_stats[stat_key] = max(v[stat_key] for v in stats_episodes.values())

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        for k, v in max_stats.items():
            logger.info(f"Max episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()
