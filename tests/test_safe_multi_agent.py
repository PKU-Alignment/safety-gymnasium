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
"""Test multi-agent environments."""

import safety_gymnasium


scenario_agent_conf = {
    'Ant': ['2x4', '4x2'],
    'HalfCheetah': ['6x1', '2x3'],
    'Hopper': ['3x1'],
    'Humanoid': ['9|8'],
    'Swimmer': ['2x1'],
    'Walker2d': ['2x3'],
}


def test_safe_ma():
    """Test env."""
    for scenario, agent_confs in scenario_agent_conf.items():
        for agent_conf in agent_confs:
            env = safety_gymnasium.make_ma(scenario, agent_conf)
            obs, _ = env.reset()
            # Use below to specify seed.
            # obs, _ = env.reset(seed=0)
            ep_ret, ep_cost = 0, 0
            for step in range(4):
                if step == 2:
                    print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
                    ep_ret, ep_cost = 0, 0
                    obs, _ = env.reset()

                act = {}
                for agent in env.agents:
                    assert env.observation_space(agent).contains(obs[agent])
                    act[agent] = env.action_space(agent).sample()
                    assert env.action_space(agent).contains(act[agent])

                obs, reward, cost, _, _, _ = env.step(act)

                ep_ret += reward['agent_0']
                ep_cost += cost['agent_0']
