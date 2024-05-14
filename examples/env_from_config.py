# Copyright 2022-2024 OmniSafe Team. All Rights Reserved.
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

from __future__ import annotations

import safety_gymnasium


if __name__ == '__main__':
    config = {
        'lidar_conf.max_dist': 3,
        'lidar_conf.num_bins': 16,
        'placements_conf.extents': [-1.5, -1.5, 1.5, 1.5],
        'Hazards': {
            'num': 1,
            'size': 0.7,
            'locations': [(0, 0)],
            'is_lidar_observed': True,
            'is_constrained': True,
            'keepout': 0.705,
        },
        'Goal': {
            'keepout': 0.305,
            'size': 0.3,
            'locations': [(1.1, 1.1)],
            'is_lidar_observed': True,
        },
    }

    # environment from the config
    env = safety_gymnasium.make('SafetyPointGoalBase-v0', config=config)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)

    s, i = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()
        s, a, d, t, i = env.step(action)
    print('done')
