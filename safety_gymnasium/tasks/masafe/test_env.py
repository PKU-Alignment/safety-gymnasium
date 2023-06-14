# Copyright 2022 Safety Gymnasium Team. All Rights Reserved.
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

import safety_gymnasium


env_id = [
    '2AgentAnt-v4',
    '2AgentAntDiag-v4',
    '4AgentAnt-v4',
    '2AgentHalfCheetah-v4',
    '6AgentHalfCheetah-v4',
    '3AgentHopper-v4',
    '2AgentHumanoid-v4',
    '2AgentHumanoidStandup-v4',
    '2AgentReacher-v4',
    '2AgentSwimmer-v4',
    '2AgentWalker2d-v4',
    'ManyAgentSwimmer-v0',
    'ManyAgentAnt-v0',
    'CoupledHalfCheetah-v0',
]
render_mode = ['human']


# pylint: disable-next=too-many-locals
def test_env(env_id, render_mode):
    """Test env."""
    env_name = env_id
    env = safety_gymnasium.make(env_name, render_mode=render_mode)
    obs, _ = env.reset()
    # Use below to specify seed.
    # obs, _ = env.reset(seed=0)
    terminated, truncated = False, False
    ep_ret = 0
    for _step in range(10):  # pylint: disable=unused-variable
        if terminated or truncated:
            # print(f'Episode Return: {ep_ret}')
            ep_ret = 0
            obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)

        # Use the environment's built_in max_episode_steps
        if hasattr(env, '_max_episode_steps'):  # pylint: disable=protected-access
            pass  # pylint: disable=unused-variable,protected-access
        # pylint: disable-next=unused-variable
        obs, reward, cost, terminated, truncated, info = env.step(act)
        ep_ret += reward

        env.render()


for env in env_id:
    for render in render_mode:
        test_env(env, render)
