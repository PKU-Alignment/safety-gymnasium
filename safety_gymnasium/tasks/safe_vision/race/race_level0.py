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
"""Race level 0."""

import numpy as np

from safety_gymnasium.assets.geoms import Goal
from safety_gymnasium.bases.base_task import BaseTask


class RaceLevel0(BaseTask):
    """A robot must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.num_steps = 500

        self.floor_conf.size = [17.5, 17.5, 0.1]

        self.palcement_cal_factor = 3.5
        robot_placements_width = self.palcement_cal_factor * 0.05
        robot_placements_lenth = self.palcement_cal_factor * 0.1
        center_x, center_y = self.palcement_cal_factor * -0.65, self.palcement_cal_factor * 0.3
        self.agent.placements = [
            (
                center_x - robot_placements_width / 2,
                center_y - robot_placements_lenth / 2,
                center_x + robot_placements_width / 2,
                center_y + robot_placements_lenth / 2,
            ),
        ]
        self.agent.keepout = 0

        self.mechanism_conf.continue_goal = False

        goal_config = {
            'reward_goal': 10.0,
            'keepout': 0.305,
            'size': 0.3,
            'locations': [(self.palcement_cal_factor * 0.9, self.palcement_cal_factor * 0.3)],
            'is_meshed': True,
            'mesh_name': 'flower_bush',
            'mesh_euler': [np.pi / 2, 0, 0],
            'mesh_height': 0.0,
        }
        self.reward_conf.reward_clip = 11
        self._add_geoms(Goal(**goal_config))
        self._is_load_static_geoms = True

        self.last_dist_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size
