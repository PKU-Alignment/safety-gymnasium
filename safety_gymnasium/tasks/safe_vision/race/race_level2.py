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
"""Race level 2."""

from safety_gymnasium.tasks.safe_vision.race.race_level1 import RaceLevel1


class RaceLevel2(RaceLevel1):
    """A robot must navigate a far way to a goal, while avoiding hazards."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        robot_placements_width = self.palcement_cal_factor * 0.05
        robot_placements_lenth = self.palcement_cal_factor * 0.05
        center_x, center_y = self.palcement_cal_factor * -0.65, self.palcement_cal_factor * -0.8
        self.agent.placements = [
            (
                center_x - robot_placements_width / 2,
                center_y - robot_placements_lenth / 2,
                center_x + robot_placements_width / 2,
                center_y + robot_placements_lenth / 2,
            ),
        ]
