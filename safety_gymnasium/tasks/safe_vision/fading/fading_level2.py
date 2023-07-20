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

from safety_gymnasium.tasks.safe_vision.fading.fading_level1 import FadingEasyLevel1


class FadingEasyLevel2(FadingEasyLevel1):
    """An agent must navigate to a goal.

    The goal will gradually disappear over time,
    while the agent should avoid more hazards and vases.
    Additionally, hazards will also disappear over time.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # pylint: disable=no-member

        self.placements_conf.extents = [-2, -2, 2, 2]

        self.hazards.num = 10
        self.vases.num = 10
        self.vases.is_constrained = True

        self.fadding_objects.extend([self.hazards])

    def specific_step(self):
        super().specific_step()

        for obj in self.fadding_objects:
            if hasattr(obj, 'cal_cost') and sum(obj.cal_cost().values()):
                self.set_objects_alpha(obj.name, obj.alpha)


class FadingHardLevel2(FadingEasyLevel2):
    """All objects will disappear, and more quickly."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.fadding_steps = 75

        self.fadding_objects.extend([self.vases])  # pylint: disable=no-member
