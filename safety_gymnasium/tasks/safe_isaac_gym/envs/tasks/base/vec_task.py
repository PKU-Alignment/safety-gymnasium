# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from gymnasium import spaces
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch


# VecEnv Wrapper for RL training
class VecTask:
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(
            np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf
        )
        self.act_space = spaces.Box(
            np.ones(self.num_actions) * task.franka_dof_lower_limits_tensor.cpu().numpy(),
            np.ones(self.num_actions) * task.franka_dof_upper_limits_tensor.cpu().numpy(),
        )

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = rl_device

        print('RL device: ', rl_device)

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations


# Python CPU/GPU Class
class VecTaskPython(VecTask):
    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.rl_device)
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_buf, rew_buf, cost_buf, reset_buf, _ = self.task.step(actions_tensor)

        return (
            torch.clamp(obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),
            rew_buf.to(self.rl_device),
            cost_buf.to(self.rl_device),
            reset_buf.to(self.rl_device),
            self.task.extras,
        )

    def reset(self):
        actions = 0.01 * (
            1
            - 2
            * torch.rand(
                [self.task.num_envs, self.task.num_actions],
                dtype=torch.float32,
                device=self.rl_device,
            )
        )

        # step the simulator
        obs_buf, rew_buf, cost_buf, reset_buf, _ = self.task.step(actions)

        return torch.clamp(obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
