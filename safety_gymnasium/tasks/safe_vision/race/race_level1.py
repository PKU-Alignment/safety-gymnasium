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
"""Race level 1."""

import numpy as np

from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_vision.race.race_level0 import RaceLevel0


class RaceLevel1(RaceLevel0):
    """A robot must navigate to a goal, while avoiding hazards."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        hazard_config = {
            'num': 7,
            'size': self.palcement_cal_factor * 0.075,
            'keepout': 0.165,
            'placements': [
                (
                    self.palcement_cal_factor * (-0.45 + 0.2 * i) - 0.3,
                    self.palcement_cal_factor * (0.3 - 0.09 * (-1) ** i) - 0.35,
                    self.palcement_cal_factor * (-0.45 + 0.2 * i) + 0.3,
                    self.palcement_cal_factor * (0.3 - 0.09 * (-1) ** i) + 0.35,
                )
                for i in range(7)
            ],
            'is_meshed': True,
            'mesh_name': 'bush',
            'mesh_euler': [np.pi / 2, 0, 0],
        }
        self._add_geoms(Hazards(**hazard_config))
        self.static_geoms_contact_cost = 1.0
