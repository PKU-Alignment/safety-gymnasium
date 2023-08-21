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
"""Apple."""

from dataclasses import dataclass, field

import numpy as np

from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_object import Geom


@dataclass
class Apples(Geom):  # pylint: disable=too-many-instance-attributes
    """Apples and Oranges are as same as Goal.

    While they can be instantiated more than one.
    And one can define different rewards for Apple and Orange.
    """

    name: str = 'apples'
    num: int = 0
    size: float = 0.3
    placements: list = None  # Placements where goal may appear (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.3  # Keepout radius when placing goals

    reward_apple: float = 1.0  # Sparse reward for being inside the goal area
    # Reward is distance towards goal plus a constant for being within range of goal
    # reward_distance should be positive to encourage moving towards the goal
    # if reward_distance is 0, then the reward function is sparse
    reward_distance: float = 1.0  # Dense reward multiplied by the distance moved to the goal

    color: np.array = COLOR['apple']
    alpha: float = 0.25
    group: np.array = GROUP['apple']
    is_lidar_observed: bool = True
    is_constrained: bool = False
    is_meshed: bool = False
    mesh_name: str = name[:-1]

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, self.size / 2 + 1e-2],
            'rot': rot,
            'geoms': [
                {
                    'name': self.name,
                    'size': [self.size, self.size / 2],
                    'type': 'cylinder',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': self.group,
                    'rgba': self.color * np.array([1, 1, 1, self.alpha]),
                },
            ],
        }
        if self.is_meshed:
            body['geoms'][0].update(
                {
                    'pos': np.r_[xy_pos, 0.3],
                    'type': 'mesh',
                    'mesh': self.mesh_name,
                    'material': self.mesh_name,
                    'euler': [np.pi / 2, 0, 0],
                },
            )
        return body

    @property
    def pos(self):
        """Helper to get goal position from layout"""
        return [self.engine.data.body(f'{self.name[:-1]}{i}').xpos.copy() for i in range(self.num)]
