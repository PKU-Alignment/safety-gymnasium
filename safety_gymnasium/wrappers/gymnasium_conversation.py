# Copyright 2022-2023 OmniSafe AI Team. All Rights Reserved.
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
"""Wrappers for converting between Safety-Gymnasium and Gymnasium environments."""

import gymnasium
from gymnasium.core import ActType


class SafetyGymnasium2Gymnasium(gymnasium.Wrapper):
    """A class that converts a Safety-Gymnasium environment to a Gymnasium environment."""

    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        info['cost'] = cost
        return obs, reward, terminated, truncated, info


class Gymnasium2SafetyGymnasium(gymnasium.Wrapper):
    """A class that converts a Gymnasium environment to a Safety-Gymnasium environment."""

    def step(self, action: ActType):
        obs, reward, terminated, truncated, info = super().step(action)
        cost = info['cost']
        return obs, reward, cost, terminated, truncated, info
