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
"""Wrapper for normalizing the output of an environment."""


import gymnasium
import numpy as np
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward, RunningMeanStd


class SafeNormalizeObservation(NormalizeObservation):
    """This wrapper will normalize observations as Gymnasium's NormalizeObservation wrapper does."""

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, costs, terminateds, truncateds, infos = self.env.step(action)
        obs = self.normalize(obs) if self.is_vector_env else self.normalize(np.array([obs]))[0]
        if 'final_observation' in infos:
            final_obs_slice = infos['_final_observation'] if self.is_vector_env else slice(None)
            infos['original_final_observation'] = infos['final_observation']
            infos['final_observation'][final_obs_slice] = self.normalize(
                infos['final_observation'][final_obs_slice],
            )
        return obs, rews, costs, terminateds, truncateds, infos


class SafeNormalizeReward(NormalizeReward):
    """This wrapper will normalize rewards as Gymnasium's NormalizeObservation wrapper does."""

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, costs, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma * (1 - terminateds) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, costs, terminateds, truncateds, infos


class SafeNormalizeCost(gymnasium.core.Wrapper, gymnasium.utils.RecordConstructorArgs):
    r"""This wrapper will normalize immediate costs s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and costs will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        """This wrapper will normalize immediate costs s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gymnasium.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gymnasium.Wrapper.__init__(self, env)

        self.num_envs = getattr(env, 'num_envs', 1)
        self.is_vector_env = getattr(env, 'is_vector_env', False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the costs returned."""
        obs, rews, costs, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            costs = np.array([costs])
        self.returns = self.returns * self.gamma * (1 - terminateds) + costs
        costs = self.normalize(costs)
        if not self.is_vector_env:
            costs = costs[0]
        return obs, rews, costs, terminateds, truncateds, infos

    def normalize(self, costs):
        """Normalizes the costs with the running mean costs and their variance."""
        self.return_rms.update(self.returns)
        return costs / np.sqrt(self.return_rms.var + self.epsilon)
