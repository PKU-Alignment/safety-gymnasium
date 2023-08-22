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
"""Geoms type objects."""

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.apples import Apples
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.buttons import Buttons
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.circle import Circle

# Extra geoms (immovable objects) to add to the scene
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.goal import Goal, GoalBlue, GoalRed
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.hazards import Hazards
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.oranges import Oranges
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.pillars import Pillars
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.sigwalls import Sigwalls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.walls import Walls


GEOMS_REGISTER = [
    Apples,
    Buttons,
    Circle,
    Goal,
    Hazards,
    Oranges,
    Pillars,
    Walls,
    Sigwalls,
    GoalRed,
    GoalBlue,
]
