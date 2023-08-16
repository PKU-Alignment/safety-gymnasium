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
"""Fading level 1."""

from safety_gymnasium.assets.free_geoms import Vases
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_vision.fading.fading_level0 import FadingEasyLevel0


class FadingEasyLevel1(FadingEasyLevel0):
    """An agent must navigate to a goal.

    The goal will gradually disappear over time,
    while the agent should avoid hazards and vases.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]

        self._add_geoms(Hazards(num=8, keepout=0.18))
        self._add_free_geoms(Vases(num=1, is_constrained=False))


class FadingHardLevel1(FadingEasyLevel1):
    """The goal will disappear more quickly."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.fadding_steps = 75
