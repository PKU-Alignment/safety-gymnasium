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
"""Test copy of environments."""

from copy import deepcopy

import gymnasium
import numpy as np

import helpers


@helpers.parametrize(
    agent_id=['Point'],
    env_id=['Goal'],
    level=['0'],
)
def test_equal_outcomes_branch(agent_id, env_id, level):
    """Test copyable env."""
    env_name = 'Safety' + agent_id + env_id + level + 'Gymnasium' + '-v0'
    env1 = gymnasium.make(env_name)
    obs, _ = env1.reset()

    env2 = deepcopy(env1)
    move = env1.action_space.sample()
    obs1, reward1, term1, trunc1, info1 = env1.step(move)
    obs2, reward2, term2, trunc2, info2 = env2.step(move)

    np.testing.assert_array_equal(obs1, obs2)
    assert reward1 == reward2
    assert term1 == term2
    assert trunc1 == trunc2
    assert info1 == info2

    env3 = deepcopy(env1)
    env4 = deepcopy(env2)
    move = env1.action_space.sample()
    obs1, reward1, term1, trunc1, info1 = env1.step(move)
    obs2, reward2, term2, trunc2, info2 = env2.step(move)
    obs3, reward3, term3, trunc3, info3 = env3.step(move)
    obs4, reward4, term4, trunc4, info4 = env4.step(move)

    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(obs2, obs3)
    np.testing.assert_array_equal(obs3, obs4)

    assert reward1 == reward2
    assert reward2 == reward3
    assert reward3 == reward4
    assert term1 == term2
    assert term2 == term3
    assert term3 == term4
    assert trunc1 == trunc2
    assert trunc2 == trunc3
    assert trunc3 == trunc4
    assert info1 == info2
    assert info2 == info3
    assert info3 == info4


@helpers.parametrize(
    agent_id=['Point', 'Car', 'Doggo'],
    env_id=['Goal'],
    level=['0', '2'],
)
def test_equal_outcomes_long(agent_id, env_id, level):
    """Test SafetyGymnasium2Gymnasium env."""
    env_name = 'Safety' + agent_id + env_id + level + 'Gymnasium' + '-v0'
    env1 = gymnasium.make(env_name)
    obs, _ = env1.reset()

    # get the env some steps away from the initial state just to be sure
    for _ in range(16):
        move = env1.action_space.sample()
        obs1, reward1, term1, trunc1, info1 = env1.step(move)

    env2 = deepcopy(env1)

    # the copied env should yield the same observations, reward, etc as the original env when the same steps are taken:
    for _ in range(32):
        move = env1.action_space.sample()
        obs1, reward1, term1, trunc1, info1 = env1.step(move)
        obs2, reward2, term2, trunc2, info2 = env2.step(move)

        np.testing.assert_array_equal(obs1, obs2)
        assert reward1 == reward2
        assert term1 == term2
        assert trunc1 == trunc2
        assert info1 == info2
