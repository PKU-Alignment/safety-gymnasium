# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
# ==============================================================================
"""Safety-Gymnasium Environments for Multi-Agent RL."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from gymnasium_robotics.envs.multiagent_mujoco.mujoco_multi import MultiAgentMujocoEnv

from safety_gymnasium.utils.task_utils import add_velocity_marker, clear_viewer


TASK_VELCITY_THRESHOLD = {
    'Ant': {'2x4': 2.522, '4x2': 2.418},
    'HalfCheetah': {'6x1': 2.932, '2x3': 3.227},
    'Hopper': {'3x1': 0.9613},
    'Humanoid': {'9|8': 0.58},
    'Swimmer': {'2x1': 0.04891},
    'Walker2d': {'2x3': 1.641},
}


class SafeMAEnv:
    """Multi-agent environment with safety constraints."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        scenario: str,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        agent_factorization: dict | None = None,
        local_categories: list[list[str]] | None = None,
        global_categories: tuple[str, ...] | None = None,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        assert scenario in TASK_VELCITY_THRESHOLD, f'Invalid agent: {scenario}'
        self.agent = scenario
        if agent_conf not in TASK_VELCITY_THRESHOLD[scenario]:
            vel_temp_conf = next(iter(TASK_VELCITY_THRESHOLD[scenario]))
            self._velocity_threshold = TASK_VELCITY_THRESHOLD[scenario][vel_temp_conf]
            warnings.warn(
                f'\033[93mUnknown agent configuration: {agent_conf} \033[0m'
                f'\033[93musing default velocity threshold {self._velocity_threshold} \033[0m'
                f'\033[93mfor agent {scenario} and configuration {vel_temp_conf}.\033[0m',
                UserWarning,
                stacklevel=2,
            )
        else:
            self._velocity_threshold = TASK_VELCITY_THRESHOLD[scenario][agent_conf]
        self.env: MultiAgentMujocoEnv = MultiAgentMujocoEnv(
            scenario,
            agent_conf,
            agent_obsk,
            agent_factorization,
            local_categories,
            global_categories,
            render_mode,
            **kwargs,
        )
        self.env.single_agent_env.model.light(0).castshadow = False

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith('_'):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        """Reset the environment."""
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        """Step the environment."""
        observations, rewards, terminations, truncations, info = self.env.step(action)
        info_single = info[self.env.possible_agents[0]]
        velocity = np.sqrt(info_single['x_velocity'] ** 2 + info_single.get('y_velocity', 0) ** 2)
        if self.agent == 'Swimmer':
            velocity = info_single['x_velocity']
        cost_n = float(velocity > self._velocity_threshold)
        costs = {}
        for agents in self.env.possible_agents:
            costs[agents] = cost_n

        viewer = self.env.single_agent_env.mujoco_renderer.viewer
        if viewer:
            clear_viewer(viewer)
            add_velocity_marker(
                viewer=viewer,
                pos=self.env.single_agent_env.get_body_com('torso')[:3].copy(),
                vel=velocity,
                cost=cost_n,
                velocity_threshold=self._velocity_threshold,
            )

        return observations, rewards, costs, terminations, truncations, info


make_ma = SafeMAEnv  # pylint: disable=invalid-name
