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

import numpy as np

import helpers
import safety_gymnasium
from safety_gymnasium.tasks.safe_navigation.button.button_configs import button_level0
from safety_gymnasium.tasks.safe_navigation.circle.circle_configs import circle_level0
from safety_gymnasium.tasks.safe_navigation.goal.goal_configs import goal_level0
from safety_gymnasium.tasks.safe_navigation.push.push_configs import push_level0
from safety_gymnasium.tasks.safe_navigation.run.run_configs import run_level0


@helpers.parametrize(
    environments=[
        ('SafetyPointGoal0-v0', 'SafetyPointGoalBase-v0', goal_level0),
        ('SafetyPointPush0-v0', 'SafetyPointPushBase-v0', push_level0),
        ('SafetyPointButton0-v0', 'SafetyPointButtonBase-v0', button_level0),
        ('SafetyPointRun0-v0', 'SafetyPointRunBase-v0', run_level0),
        ('SafetyPointCircle0-v0', 'SafetyPointCircleBase-v0', circle_level0),
    ],
)
def test_config(environments):
    """Test config environments"""
    env_id, base_env_id, config = environments
    # environment from the config
    env = safety_gymnasium.make(base_env_id, config=config)
    # the registered environment
    env2 = safety_gymnasium.make(env_id)
    s, _ = env.reset(seed=42)
    s2, _ = env2.reset(seed=42)
    all_done = False
    num_iter = 0
    while (not all_done) and num_iter < 4:
        action = env2.action_space.sample()
        s, a, c, done, terminated, _ = env.step(action)
        s2, a2, c2, done2, terminated2, _ = env2.step(action)
        assert np.isclose(s, s2).all()
        assert np.isclose(a, a2).all()
        assert np.isclose(c, c2).all()
        all_done = (done and done2) or (terminated and terminated2)
        num_iter += 1
