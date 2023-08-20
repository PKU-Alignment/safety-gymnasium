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
"""Doggo."""

from __future__ import annotations

from safety_gymnasium.bases.base_agent import BaseAgent
from safety_gymnasium.utils.random_generator import RandomGenerator


class Doggo(BaseAgent):
    """A quadrupedal robot with bilateral symmetry.

    Each of the four legs has two controls at the hip,
    for azimuth and elevation relative to the torso,
    and one in the knee controlling angle.
    It is designed such that a uniform random policy should keep
    the robot from falling over and generate some travel.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        random_generator: RandomGenerator,
        placements: list | None = None,
        locations: list | None = None,
        keepout: float = 0.4,
        rot: float | None = None,
    ) -> None:
        super().__init__(
            self.__class__.__name__,
            random_generator,
            placements,
            locations,
            keepout,
            rot,
        )
        self.sensor_conf.sensors += (
            'touch_ankle_1a',
            'touch_ankle_2a',
            'touch_ankle_3a',
            'touch_ankle_4a',
            'touch_ankle_1b',
            'touch_ankle_2b',
            'touch_ankle_3b',
            'touch_ankle_4b',
        )

    def is_alive(self):
        """Doggo runs until timeout."""
        return True

    def reset(self):
        """No need to reset anything."""
