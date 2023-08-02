from typing import Optional, Tuple, List

import torch
from torch import nn as nn
from torch import optim as optim

from habitat_baselines.rl.ppo.ppo import PPO
from habitat.utils import profiling_wrapper
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin

from habitat_scanbot2d.auxiliary_tasks.rollout_auxiliary_tasks import (
    RolloutAuxiliaryTask,
)
from habitat_scanbot2d.auxiliary_tasks.auxiliary_rollout_storage import (
    AuxiliaryRoulloutStorage,
)


class AuxiliaryPPO(PPO):
    def __init__(
        self,
        actor_critic: Policy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
        use_normalized_advantage: bool = True,
        auxiliary_tasks: List[RolloutAuxiliaryTask] = [],
        auxiliary_loss_coef: Optional[float] = None,
    ) -> None:
        super().__init__(
            actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr,
            eps,
            max_grad_norm,
            use_clipped_value_loss,
            use_normalized_advantage,
        )
        self.auxiliary_tasks = auxiliary_tasks
        self.auxiliary_loss_coef = auxiliary_loss_coef
        params = list(self.actor_critic.parameters())
        for task in auxiliary_tasks:
            params += list(task.parameters())
        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, params)), lr=lr, eps=eps
        )

    def update(
        self, rollouts: AuxiliaryRoulloutStorage
    ) -> Tuple[float, float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        auxiliary_task_loss_epoch = 0.0

        for _e in range(self.ppo_epoch):
            profiling_wrapper.range_push("AuxiliaryPPO.update epoch")
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for batch in data_generator:
                (values, action_log_probs, dist_entropy, _,) = self._evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["masks"],
                    batch["actions"],
                )

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * batch["advantages"]
                )
                action_loss = -(torch.min(surr1, surr2).mean())

                if self.use_clipped_value_loss:
                    value_pred_clipped = batch["value_preds"] + (
                        values - batch["value_preds"]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch["returns"]).pow(2)
                    value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(
                        2
                    )
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
                else:
                    value_loss = 0.5 * (batch["returns"] - values).pow(2)

                value_loss = value_loss.mean()
                dist_entropy = dist_entropy.mean()

                aux_losses = torch.tensor([])
                if len(self.auxiliary_tasks) > 0:
                    raw_aux_loss = self.actor_critic.compute_auxiliary_task_losses(
                        batch, rollouts.numsteps,
                    )
                    aux_losses = torch.stack(raw_aux_loss)
                total_aux_loss = torch.sum(aux_losses, dim=0)

                self.optimizer.zero_grad()
                total_loss = (
                    action_loss
                    + value_loss * self.value_loss_coef
                    + total_aux_loss * self.auxiliary_loss_coef
                    - dist_entropy * self.entropy_coef
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                auxiliary_task_loss_epoch += total_aux_loss.item()

            profiling_wrapper.range_pop()  # AuxiliaryPPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        auxiliary_task_loss_epoch /= num_updates

        return (
            value_loss_epoch,
            action_loss_epoch,
            dist_entropy_epoch,
            auxiliary_task_loss_epoch,
        )


class AuxiliaryDDPPO(DecentralizedDistributedMixin, AuxiliaryPPO):
    pass
