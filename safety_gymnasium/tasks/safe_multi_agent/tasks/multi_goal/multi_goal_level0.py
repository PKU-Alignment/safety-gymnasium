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
"""Multi Goal level 0."""

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.goal import GoalBlue, GoalRed
from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask


class MultiGoalLevel0(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1, -1, 1, 1]

        self._add_geoms(
            GoalRed(keepout=0.305),
            GoalBlue(keepout=0.305),
        )

        self.last_dist_goal_red = None
        self.last_dist_goal_blue = None

    def dist_goal_red(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal_red'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(0, self.goal_red.pos)  # pylint: disable=no-member

    def dist_goal_blue(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal_blue'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(1, self.goal_blue.pos)  # pylint: disable=no-member

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = {'agent_0': 0.0, 'agent_1': 0.0}

        dist_goal_red = self.dist_goal_red()
        reward['agent_0'] += (
            self.last_dist_goal_red - dist_goal_red
        ) * self.goal_red.reward_distance
        self.last_dist_goal_red = dist_goal_red

        if self.goal_achieved[0]:
            reward['agent_0'] += self.goal_red.reward_goal

        dist_goal_blue = self.dist_goal_blue()
        reward['agent_1'] += (
            self.last_dist_goal_blue - dist_goal_blue
        ) * self.goal_blue.reward_distance
        self.last_dist_goal_blue = dist_goal_blue

        if self.goal_achieved[1]:
            reward['agent_1'] += self.goal_blue.reward_goal

        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal_red = self.dist_goal_red()
        self.last_dist_goal_blue = self.dist_goal_blue()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable=no-member
        return (
            self.dist_goal_red() <= self.goal_red.size,
            self.dist_goal_blue() <= self.goal_blue.size,
        )
