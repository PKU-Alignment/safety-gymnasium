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

import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler


class RolloutStorage:
    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        states_shape,
        actions_shape,
        device='cpu',
        sampler='sequential',
    ) -> None:
        self.device = device
        self.sampler = sampler

        # Core
        self.observations = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *obs_shape,
            device=self.device,
        )
        self.states = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *states_shape,
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.costs = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
        )
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        # For PPO
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
        )
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
        )

        # For cost
        self.cost_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.creturns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.cadvantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(
        self,
        observations,
        states,
        actions,
        rewards,
        costs,
        dones,
        values,
        cost_values,
        actions_log_prob,
        mu,
        sigma,
    ):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError('Rollout buffer overflow')

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.costs[self.step].copy_(costs.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.cost_values[self.step].copy_(cost_values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def compute_costs(self, last_cost_values, gamma, lam):
        cadvantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_cvalues = last_cost_values
            else:
                next_cvalues = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.costs[step]
                + next_is_not_terminal * gamma * next_cvalues
                - self.cost_values[step]
            )
            cadvantage = delta + next_is_not_terminal * gamma * lam * cadvantage
            self.creturns[step] = cadvantage + self.cost_values[step]

        # Compute and normalize the cadvantages
        self.cadvantages = self.creturns - self.cost_values
        self.cadvantages = (self.cadvantages - self.cadvantages.mean()) / (
            self.cadvantages.std() + 1e-8
        )

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            ),
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean(), self.costs.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == 'sequential':
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == 'random':
            subset = SubsetRandomSampler(range(batch_size))

        return BatchSampler(subset, mini_batch_size, drop_last=True)
