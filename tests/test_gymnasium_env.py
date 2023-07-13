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
"""Test API conversion."""

import gymnasium

import helpers
import safety_gymnasium


@helpers.parametrize(
    agent_id=['Point'],
    env_id=['Goal'],
    level=['0'],
)
def test_navigation_env(agent_id, env_id, level):
    """Test SafetyGymnasium2Gymnasium env."""
    env_name = 'Safety' + agent_id + env_id + level + 'Gymnasium' + '-v0'
    env = gymnasium.make(env_name)
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

        obs, reward, _, _, info = env.step(act)
        ep_ret += reward
        ep_cost += info['cost']


@helpers.parametrize(
    agent_id=['Point'],
    env_id=['Goal'],
    level=['0'],
)
def test_convert_api(agent_id, env_id, level):
    """Test Gymnasium2SafetyGymnasium env."""
    env_name = 'Safety' + agent_id + env_id + level + 'Gymnasium' + '-v0'
    env = gymnasium.make(env_name)
    env = gymnasium.wrappers.ClipAction(env)
    env = safety_gymnasium.wrappers.Gymnasium2SafetyGymnasium(env)
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
    agent_id=['Point'],
    env_id=['Goal'],
    level=['0'],
)
def test_with_wrappers(agent_id, env_id, level):
    """Test with_wrappers use case."""
    env_name = 'Safety' + agent_id + env_id + level + '-v0'
    env = safety_gymnasium.wrappers.with_gymnasium_wrappers(
        safety_gymnasium.make(env_name),
        gymnasium.wrappers.ClipAction,
    )
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
