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
"""Run with a custom config."""

import numpy as np

from safety_gymnasium.bases.base_task import BaseTask


class RunBase(BaseTask):
    """An agent must run as far as possible while avoid going outside the boundary."""

    def __init__(self, config) -> None:
        assert 'Sigwalls' in config, 'config must have the field `Sigwalls`'
        self.reward_factor: float = 1.0
        super().__init__(config=config)

        self.old_potential = None

    def calculate_reward(self):
        """The agent should run as far as possible."""
        reward = 0.0
        potential = -np.linalg.norm(self.agent.pos[:2] - self.goal_pos) * self.reward_factor
        reward += potential - self.old_potential
        self.old_potential = potential
        return reward

    def specific_reset(self):
        self.old_potential = (
            -np.linalg.norm(self.agent.pos[:2] - self.goal_pos) * self.reward_factor
        )

    def specific_step(self):
        pass

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        """Weather the goal of task is achieved."""
        return False

    @property
    def goal_pos(self):
        """Fixed goal position."""
        return [[0, -1e3]]
