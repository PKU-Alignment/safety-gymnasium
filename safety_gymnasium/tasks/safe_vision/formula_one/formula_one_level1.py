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
"""FormulaOne level 1."""

from safety_gymnasium.assets.free_geoms import Vases
from safety_gymnasium.tasks.safe_vision.formula_one.formula_one_level0 import FormulaOneLevel0


class FormulaOneLevel1(FormulaOneLevel0):
    """Barricades are added into the map, and the agent must avoid them.

    Meanwhile, the agent must still navigate to the goal and further avoid hitting the walls.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        vases_placements = [(-12.2, -32.5, 27.8, 7.5), (26, -11, 38, 1)]
        vases_config = {
            'num': 200,
            'keepout': 0.5,
            'placements': vases_placements,
            'velocity_cost': 0,
            'is_meshed': True,
            'mesh_name': 'road_barrier',
        }
        self._add_free_geoms(Vases(**vases_config))
        self.static_geoms_contact_cost = 1.0
