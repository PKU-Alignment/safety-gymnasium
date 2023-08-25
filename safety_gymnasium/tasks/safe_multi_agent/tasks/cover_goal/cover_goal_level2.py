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
"""CoverGoal level 2."""

from safety_gymnasium.tasks.safe_multi_agent.tasks.cover_goal.cover_goal_level1 import (
    CoverGoalLevel1,
)


class CoverGoalLevel2(CoverGoalLevel1):

    def __init__(self, config, agent_num) -> None:
        super().__init__(config=config, agent_num=agent_num)
        # pylint: disable=no-member

        self.placements_conf.extents = [-2, -2, 2, 2]

        self.hazards.num = 10
        self.vases.num = 10
        self.vases.is_constrained = True
        self.contact_other_cost = 1.0
