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
"""Staged Goal."""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from safety_gymnasium.assets.geoms.goal import Goal


@dataclass
class StagedGoal(Goal):  # pylint: disable=too-many-instance-attributes
    """A specific goal which allows for multiple stages.

    The agent must navigate to each goal in the `staged_locations` list in order.
    """

    name: str = 'staged_goal'
    num_stage: int = 1
    staged_locations: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post init."""
        self._goal_idx: int = 0
        assert self.num_stage == len(
            self.staged_locations,
        ), 'Number of staged locations must match number of goals'

    def reset(self, agent_pos: np.ndarray):
        """Reset the location according to agent."""
        idx = self.find_pair_index(agent_pos, self.staged_locations)
        if idx == -1:
            print('Pair not found in list!')
            self._goal_idx = 0
        else:
            # Ensure that index doesn't exceed the list length
            self._goal_idx = (idx + 1) % len(self.staged_locations)

    def get_next_goal_xy(self):
        """Switch to next goal."""
        assert self._goal_idx < self.num_stage, 'No more goals to stage.'
        goal_xy = self.staged_locations[self._goal_idx]
        self._goal_idx = (self._goal_idx + 1) % self.num_stage
        return goal_xy

    def is_approx_equal(self, num1, num2, epsilon=1e-6):
        """Check if two numbers are approximately equal."""
        return abs(num1 - num2) < epsilon

    def find_pair_index(self, pair, pair_list):
        """Find index of pair in list of pairs."""
        for idx, (i, j) in enumerate(pair_list):
            if self.is_approx_equal(i, pair[0]) and self.is_approx_equal(j, pair[1]):
                return idx
        return -1
