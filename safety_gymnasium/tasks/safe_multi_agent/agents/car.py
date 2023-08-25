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
"""Car."""

from typing import Optional

import glfw
import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.bases.base_agent import BaseAgent
from safety_gymnasium.tasks.safe_multi_agent.utils.random_generator import RandomGenerator


class Car(BaseAgent):
    """Car is a slightly more complex agent.

    Which has two independently-driven parallel wheels and a free rolling rear wheel.
    Car is not fixed to the 2D-plane, but mostly resides in it.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        random_generator: RandomGenerator,
        placements: Optional[list] = None,
        locations: Optional[list] = None,
        keepout: float = 0.4,
        rot: Optional[float] = None,
        num: int = 2,
    ) -> None:
        self.actuator_index = np.array([i for i in range(2)])
        self.delta = 2
        super().__init__(
            self.__class__.__name__,
            random_generator,
            placements,
            locations,
            keepout,
            rot,
            num,
        )

    def is_alive(self):
        """Point runs until timeout."""
        return True

    def reset(self):
        """No need to reset anything."""

    def debug(self):
        """Apply action which inputted from keyboard."""
        action = np.zeros(2 * self.num)
        for key in self.debug_info.keys:
            if key == glfw.KEY_I:
                action[:2] += np.array([1, 1])
            elif key == glfw.KEY_K:
                action[:2] += np.array([-1, -1])
            elif key == glfw.KEY_J:
                action[:2] = np.array([1, -1])
                break
            elif key == glfw.KEY_L:
                action[:2] = np.array([-1, 1])
                break
        self.apply_action(action)
