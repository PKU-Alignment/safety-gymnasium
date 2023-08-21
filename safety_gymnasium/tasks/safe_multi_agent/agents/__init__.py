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
"""Agents."""

from safety_gymnasium.tasks.safe_multi_agent.agents.ant import Ant
from safety_gymnasium.tasks.safe_multi_agent.agents.car import Car
from safety_gymnasium.tasks.safe_multi_agent.agents.doggo import Doggo
from safety_gymnasium.tasks.safe_multi_agent.agents.point import Point
from safety_gymnasium.tasks.safe_multi_agent.agents.racecar import Racecar


Registry = {
    'ant': Ant,
    'car': Car,
    'point': Point,
    'racecar': Racecar,
    'doggo': Doggo,
}
