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
"""FormulaOne level 0."""

from safety_gymnasium.assets.geoms.staged_goal import StagedGoal
from safety_gymnasium.bases.base_task import BaseTask


class FormulaOneLevel0(BaseTask):
    """A robot must navigate to a goal in the Formula One map.

    While the goal is divided as 7 stages.
    And the agent can get reward only when it reaches the goal of each stage.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.num_steps = 1000

        self.floor_conf.size = [0.5, 0.5, 0.1]

        staged_points = [
            (3, 9),
            (13, -1.7),
            (26, 0.05),
            (32, -7),
            (4, -17.5),
            (19.0, -20.7),
            (-0.85, -0.4),
        ]

        delta = 1e-9
        self.agent.placements = [
            (x - delta, y - delta, x + delta, y + delta) for x, y in staged_points
        ]
        self.agent.keepout = 0.0

        self.mechanism_conf.continue_goal = True

        goal_config = {
            'reward_goal': 10.0,
            'keepout': 0.305,
            'size': 0.3,
            'is_meshed': True,
        }
        self.reward_conf.reward_clip = 11
        self._add_geoms(StagedGoal(num_stage=7, staged_locations=staged_points, **goal_config))
        self._is_load_static_geoms = True

        self.last_dist_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_staged_goal()
        reward += (self.last_dist_goal - dist_goal) * self.staged_goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.staged_goal.reward_goal

        return reward

    def specific_reset(self):
        self.staged_goal.reset(self.agent.pos)  # pylint: disable=no-member

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_staged_goal_position()
        self.last_dist_goal = self.dist_staged_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_staged_goal() <= self.staged_goal.size
