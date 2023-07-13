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
"""Test environments."""

import helpers
import safety_gymnasium


@helpers.parametrize(
    agent_id=['Point', 'Car', 'Racecar', 'Ant', 'Doggo'],
    env_id=['Goal', 'Push', 'Button', 'Circle'],
    level=['0', '1', '2'],
    render_mode=['rgb_array', 'depth_array'],
)
def test_env_render(agent_id, env_id, level, render_mode):
    """Test env."""
    env_name = 'Safety' + agent_id + env_id + level + '-v0'
    env = safety_gymnasium.make(env_name, render_mode=render_mode)
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

        env.render()


@helpers.parametrize(
    agent_id=['Point', 'Car', 'Racecar', 'Ant', 'Doggo'],
    env_id=['Run'],
    level=['0'],
    render_mode=['rgb_array', 'depth_array'],
)
def test_run_env_render(agent_id, env_id, level, render_mode):
    """Test env."""
    env_name = 'Safety' + agent_id + env_id + level + '-v0'
    env = safety_gymnasium.make(env_name, render_mode=render_mode)
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

        # Use the environment's built_in max_episode_steps
        if hasattr(env, '_max_episode_steps'):
            pass
        obs, reward, cost, _, _, _ = env.step(act)
        ep_ret += reward
        ep_cost += cost

        env.render()


@helpers.parametrize(
    agent_id=['Humanoid', 'Ant', 'Hopper', 'HalfCheetah', 'Swimmer', 'Walker2d'],
    env_id=['Velocity'],
    render_mode=['rgb_array', 'depth_array'],
    version=['v0', 'v1'],
)
def test_velocity_env_render(agent_id, env_id, render_mode, version):
    """Test env."""
    env_name = 'Safety' + agent_id + env_id + '-' + version
    env = safety_gymnasium.make(env_name, render_mode=render_mode)
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

        # Use the environment's built_in max_episode_steps
        if hasattr(env, '_max_episode_steps'):
            pass
        obs, reward, cost, _, _, _ = env.step(act)
        ep_ret += reward
        ep_cost += cost

        env.render()


@helpers.parametrize(
    agent_id=['Point'],
    env_id=['Goal'],
    level=['0'],
    render_mode=['rgb_array_list', 'depth_array_list'],
)
def test_env_render_list(agent_id, env_id, level, render_mode):
    """Test env."""
    env_name = 'Safety' + agent_id + env_id + level + '-v0'
    env = safety_gymnasium.make(env_name, render_mode=render_mode)
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

        # Use the environment's built_in max_episode_steps
        if hasattr(env, '_max_episode_steps'):
            pass
        obs, reward, cost, _, _, _ = env.step(act)
        ep_ret += reward
        ep_cost += cost

    assert isinstance(env.render(), list)


@helpers.parametrize(
    agent_id=['Humanoid'],
    env_id=['Velocity'],
    render_mode=['rgb_array_list', 'depth_array_list'],
    version=['v0', 'v1'],
)
def test_velocity_env_render_list(agent_id, env_id, render_mode, version):
    """Test env."""
    env_name = 'Safety' + agent_id + env_id + '-' + version
    env = safety_gymnasium.make(env_name, render_mode=render_mode)
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

        # Use the environment's built_in max_episode_steps
        if hasattr(env, '_max_episode_steps'):
            pass
        obs, reward, cost, _, _, _ = env.step(act)
        ep_ret += reward
        ep_cost += cost

    assert isinstance(env.render(), list)
