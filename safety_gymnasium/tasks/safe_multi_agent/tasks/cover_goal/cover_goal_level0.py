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
"""CoverGoal level 0."""

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.goal import Goals
from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask
import numpy as np

class CoverGoalLevel0(BaseTask):

    def __init__(self, config, agent_num) -> None:
        super().__init__(config=config, agent_num=agent_num)

        self.placements_conf.extents = [-1, -1, 1, 1]

        self._add_geoms(
            Goals(keepout=0.305, num=self.agents.num),
        )
        self.goal_achieved_index = np.zeros(self.agents.num, dtype=bool)

    def dist_index_goals(self, index) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goals'), 'Please make sure you have added goal into env.'
        return [self.agents.dist_xy(pos, index) for pos in self.goals.pos]

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = {f'agent_{i}': 0.0 for i in range(self.agents.num)}

        if self.goal_achieved.all():
            for index in range(self.agents.num):
                reward[f'agent_{index}'] += self.goals.reward_goal

        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        self.build_goals_position()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable=no-member
        self.goal_achieved_index = np.zeros(self.agents.num, dtype=bool)

        for index in range(self.agents.num):
            dist_goal = np.array(self.dist_index_goals(index))
            local_achieved = dist_goal <= self.goals.size
            self.goal_achieved_index |= local_achieved


        return self.goal_achieved_index.all()
