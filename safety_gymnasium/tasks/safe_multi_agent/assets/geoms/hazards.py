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
"""Hazard."""

from dataclasses import dataclass, field

import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.assets.color import COLOR
from safety_gymnasium.tasks.safe_multi_agent.assets.group import GROUP
from safety_gymnasium.tasks.safe_multi_agent.bases.base_object import Geom


@dataclass
class Hazards(Geom):  # pylint: disable=too-many-instance-attributes
    """Hazardous areas."""

    name: str = 'hazards'
    num: int = 0  # Number of hazards in an environment
    size: float = 0.2
    placements: list = None  # Placements list for hazards (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.4  # Radius of hazard keepout for placement
    alpha: float = COLOR['hazard'][-1]
    cost: float = 1.0  # Cost (per step) for violating the constraint

    color: np.array = COLOR['hazard']
    group: np.array = GROUP['hazard']
    is_lidar_observed: bool = True
    is_constrained: bool = True
    is_meshed: bool = False

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        geom = {
            'name': self.name,
            'size': [self.size, 1e-2],  # self.hazards_size / 2],
            'pos': np.r_[xy_pos, 2e-2],  # self.hazards_size / 2 + 1e-2],
            'rot': rot,
            'type': 'cylinder',
            'contype': 0,
            'conaffinity': 0,
            'group': self.group,
            'rgba': self.color,
        }
        if self.is_meshed:
            geom.update(
                {
                    'type': 'mesh',
                    'mesh': 'bush',
                    'material': 'bush',
                    'euler': [np.pi / 2, 0, 0],
                },
            )
        return geom

    def cal_cost(self):
        """Contacts Processing."""
        cost = {'agent_0': {}, 'agent_1': {}}
        if not self.is_constrained:
            return cost
        cost_0 = cost['agent_0']
        cost_0['cost_hazards'] = 0
        for h_pos in self.pos:
            h_dist = self.agent.dist_xy(0, h_pos)
            # pylint: disable=no-member
            if h_dist <= self.size:
                cost_0['cost_hazards'] += self.cost * (self.size - h_dist)

        cost_1 = cost['agent_1']
        cost_1['cost_hazards'] = 0
        for h_pos in self.pos:
            h_dist = self.agent.dist_xy(1, h_pos)
            # pylint: disable=no-member
            if h_dist <= self.size:
                cost_1['cost_hazards'] += self.cost * (self.size - h_dist)

        return cost

    @property
    def pos(self):
        """Helper to get the hazards positions from layout."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'hazard{i}').xpos.copy() for i in range(self.num)]
