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
"""Test vision environments."""

import helpers
import safety_gymnasium


@helpers.parametrize(
    agent_id=['Point', 'Car', 'Racecar', 'Ant'],
    env_id=[
        'Goal',
        'Push',
        'Button',
        'Race',
        'FadingEasy',
        'FadingHard',
        'BuildingGoal',
        'BuildingPush',
        'BuildingButton',
        'FormulaOne',
    ],
    level=['0', '1', '2'],
)
def test_vision_env(agent_id, env_id, level):
    """Test vision env."""
    env_name = 'Safety' + agent_id + env_id + level + 'Vision' + '-v0'
    env = safety_gymnasium.make(env_name)
    obs, _ = env.reset()
    ep_ret, ep_cost = 0, 0
    for step in range(4):
        if step == 2:
            print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
            ep_ret, ep_cost = 0, 0
            obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        obs, reward, cost, _, _, _ = env.step(act)

        ep_ret += reward
        ep_cost += cost


@helpers.parametrize(
    agent_id=['Point', 'Car', 'Racecar', 'Ant'],
    env_id=['Run'],
    level=['0'],
)
def test_new_env(agent_id, env_id, level):
    """Test env."""
    env_name = 'Safety' + agent_id + env_id + level + 'Vision' + '-v0'
    env = safety_gymnasium.make(env_name)
    obs, _ = env.reset()
    ep_ret, ep_cost = 0, 0
    for step in range(4):
        if step == 2:
            print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
            ep_ret, ep_cost = 0, 0
            obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)

        obs, reward, cost, _, _, _ = env.step(act)

        ep_ret += reward
        ep_cost += cost
