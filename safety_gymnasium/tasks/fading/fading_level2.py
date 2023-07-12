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
"""Fading level 2."""

from safety_gymnasium.tasks.fading.fading_level1 import FadingLevel1


class FadingLevel2(FadingLevel1):
    """An agent must navigate to a goal.

    The goal will gradually disappear over time,
    while avoiding more hazards and vases that will also disappear over time.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # pylint: disable=no-member

        self.placements_conf.extents = [-2, -2, 2, 2]

        self.hazards.num = 10
        self.vases.num = 10
        self.vases.is_constrained = True

        self.fadding_objects.extend([self.vases, self.hazards])

    def specific_step(self):
        super().specific_step()

        # pylint: disable=no-member
        if sum(self.vases.cal_cost().values()):
            self.set_objects_alpha(self.vases.name, self.vases.alpha)
        if sum(self.hazards.cal_cost().values()):
            self.set_objects_alpha(self.hazards.name, self.hazards.alpha)
        # pylint: enable=no-member
