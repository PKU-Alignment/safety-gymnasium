# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import statistics
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.storage import RolloutStorage
from gymnasium.spaces import Space
from torch.nn.functional import softplus
from torch.utils.tensorboard import SummaryWriter


class FOCOPS:
    def __init__(
        self,
        vec_env,
        logger,
        actor_class,
        critic_class,
        cost_critic_class,
        num_transitions_per_env,
        num_learning_epochs,
        num_mini_batches,
        cost_lim,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        init_noise_std=1.0,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=0.5,
        use_clipped_value_loss=True,
        schedule='fixed',
        desired_kl=None,
        model_cfg=None,
        device='cpu',
        sampler='sequential',
        log_dir='run',
        is_testing=False,
        print_log=True,
        apply_reset=False,
        asymmetric=False,
    ) -> None:
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError('vec_env.observation_space must be a gym Space')
        if not isinstance(vec_env.state_space, Space):
            raise TypeError('vec_env.state_space must be a gym Space')
        if not isinstance(vec_env.action_space, Space):
            raise TypeError('vec_env.action_space must be a gym Space')
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.device = device
        self.asymmetric = asymmetric
        self.cost_lim = cost_lim
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.step_size = learning_rate
        self.logger = logger

        # PPO components
        self.vec_env = vec_env

        self.actor = actor_class(
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            init_noise_std,
            model_cfg,
            asymmetric=asymmetric,
        )
        self.actor.to(self.device)

        self.critic = critic_class(
            self.observation_space.shape,
            self.state_space.shape,
            model_cfg,
            asymmetric=asymmetric,
        )
        self.critic.to(self.device)

        self.cost_critic = cost_critic_class(
            self.observation_space.shape,
            self.state_space.shape,
            model_cfg,
            asymmetric=asymmetric,
        )
        self.cost_critic.to(self.device)
        self.storage = RolloutStorage(
            self.vec_env.num_envs,
            num_transitions_per_env,
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            self.device,
            sampler,
        )
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=learning_rate)

        # PPO Lagrangian
        self.penalty_param = torch.tensor(0.0001, requires_grad=True).float()
        self.penalty = softplus(self.penalty_param)
        self.penalty_optimizer = optim.Adam([self.penalty_param], lr=0.05)

        # FOCOPS parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.focops_lam = 1.5
        self.eta = 0.02
        self.nu_lr = 0.01
        self.nu_max = 2.0
        self.nu = 0
        # print("lam", self.lam)
        # exit(0)
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset

    def test(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def load(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split('_')[-1].split('.')[0])
        self.actor.train()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        if self.is_testing:
            while True:
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                    # Compute the action
                    actions = self.actor_critic.act_inference(current_obs)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    current_obs.copy_(next_obs)
        else:
            rewbuffer = deque(maxlen=100)
            rewbuffer.append(0)
            costbuffer = deque(maxlen=100)
            costbuffer.append(0)
            lenbuffer = deque(maxlen=100)
            lenbuffer.append(0)
            cur_reward_sum = torch.zeros(
                self.vec_env.num_envs,
                dtype=torch.float,
                device=self.device,
            )
            cur_cost_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(
                self.vec_env.num_envs,
                dtype=torch.float,
                device=self.device,
            )

            reward_sum = []
            cost_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                ep_cost = []
                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    actions, actions_log_prob, mu, sigma = self.actor.act(
                        current_obs,
                        current_states,
                    )
                    values = self.critic.act(current_obs, current_states)
                    cost_values = self.cost_critic.act(current_obs, current_states)

                    # Step the vec_environment
                    next_obs, rews, costs, dones, infos = self.vec_env.step(actions)
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(
                        current_obs,
                        current_states,
                        actions,
                        rews,
                        costs,
                        dones,
                        values,
                        cost_values,
                        actions_log_prob,
                        mu,
                        sigma,
                    )
                    current_obs.copy_(next_obs)
                    current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_cost_sum[:] += costs
                        # batch_cost.append(costs.mean().cpu().numpy())
                        cur_episode_length[:] += 1
                        # print(dones)
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        cost_sum.extend(cur_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                        ep_cost.extend(cur_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                        # print(ep_cost)
                        # exit(0)
                        episode_length.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist(),
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_cost_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                # avg_cost = 0.0

                if len(ep_cost) == 0:
                    pass
                else:
                    self.avg_cost = np.mean(ep_cost) - self.cost_lim
                    self.nu += self.nu_lr * self.avg_cost
                    if self.nu < 0:
                        self.nu = 0
                    elif self.nu > self.nu_max:
                        self.nu = self.nu_max

                if self.print_log:
                    rewbuffer.extend(reward_sum)
                    costbuffer.extend(cost_sum)
                    lenbuffer.extend(episode_length)

                last_values = self.critic.act(current_obs, current_states)
                last_cost_values = self.cost_critic.act(current_obs, current_states)
                stop = time.time()
                stop - start

                mean_trajectory_length, mean_reward, mean_cost = self.storage.get_statistics()
                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                self.storage.compute_costs(last_cost_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                stop - start
                self.writer.add_scalar('Train/mean_reward', statistics.mean(rewbuffer), it)
                self.writer.add_scalar('Train/mean_cost', statistics.mean(costbuffer), it)
                self.logger.store(Reward=statistics.mean(rewbuffer))
                self.logger.store(Cost=statistics.mean(costbuffer))
                self.logger.store(Epoch=it)
                self.logger.log_tabular('Epoch', average_only=True)
                self.logger.log_tabular('Reward', average_only=True)
                self.logger.log_tabular('Cost', average_only=True)
                self.logger.dump_tabular()

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for _epoch in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch = self.storage.observations.view(
                    -1,
                    *self.storage.observations.size()[2:],
                )[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[
                        indices
                    ]
                else:
                    states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[
                    indices
                ]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                # For cost
                cost_target_values_batch = self.storage.cost_values.view(-1, 1)[indices]

                returns_batch = self.storage.returns.view(-1, 1)[indices]
                # For cost
                cost_returns_batch = self.storage.creturns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]

                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                # For cost
                cadvantages_batch = self.storage.cadvantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[
                    indices
                ]

                actions_log_prob_batch, entropy_batch, mu_batch, sigma_batch = self.actor.evaluate(
                    obs_batch,
                    states_batch,
                    actions_batch,
                )
                value_batch = self.critic.evaluate(obs_batch, states_batch)
                cost_value_batch = self.cost_critic.evaluate(obs_batch, states_batch)

                # KL
                if self.desired_kl is not None and self.schedule == 'adaptive':
                    kl = torch.sum(
                        sigma_batch
                        - old_sigma_batch
                        + (
                            torch.square(old_sigma_batch.exp())
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch.exp()))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size

                # Surrogate loss
                ratio = torch.exp(
                    actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch).detach(),
                )
                surrogate = torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = torch.squeeze(advantages_batch) * torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                )
                surrogate_loss = torch.min(surrogate, surrogate_clipped).mean()

                # Surrogate cost
                (torch.squeeze(cadvantages_batch) * ratio).mean()
                # surrogate_cost_clipped = torch.squeeze(cadvantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                #                                                                    1.0 + self.clip_param)
                # surrogate_cost_loss = torch.min(surrogate_cost, surrogate_cost_clipped).mean()

                """
                pi_objective = loss_r - self.penalty_item * loss_c
                pi_objective = pi_objective / (1 + self.penalty_item)
                loss_pi = -pi_objective
                """

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param,
                        self.clip_param,
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                for param in self.critic.parameters():
                    value_loss += param.pow(2).sum() * 0.001
                # value critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # cost function loss
                if self.use_clipped_value_loss:
                    cost_value_clipped = cost_target_values_batch + (
                        cost_value_batch - cost_target_values_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                    cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                    cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
                else:
                    cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

                # value critic
                for param in self.cost_critic.parameters():
                    cost_value_loss += param.pow(2).sum() * 0.001
                self.cost_critic_optimizer.zero_grad()
                cost_value_loss.backward()
                nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
                self.cost_critic_optimizer.step()

                # self.penalty_item = softplus(self.penalty_param).detach()
                # loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()
                # print("surrogate_loss", surrogate_cost)
                # print("surro")

                # print("self.penalty_item", self.penalty_param)
                # self.penalty_item = softplus(self.penalty_param).item()
                # exit(0)
                # loss = -surrogate_loss + self.penalty_item * surrogate_cost_loss - self.entropy_coef * entropy_batch.mean()
                loss = (
                    kl_mean
                    - (1 / self.focops_lam)
                    * ratio
                    * (torch.squeeze(advantages_batch) - self.nu * torch.squeeze(cadvantages_batch))
                ) * (kl_mean.detach() <= self.eta).type(torch.float32)
                loss = loss.mean()
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss
