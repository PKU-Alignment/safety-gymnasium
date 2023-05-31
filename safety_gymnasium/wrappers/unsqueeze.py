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
"""Wrapper for unsqueezing the output of an environment."""


from __future__ import annotations

import gymnasium
import numpy as np


class SafeUnsqueeze(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    """Unsqueeze the observation, reward, cost, terminated, truncated and info.

    Examples:
        >>> env = UnsqueezeWrapper(env)
    """

    def __init__(self, env: gymnasium.Env) -> None:
        """Initialize an instance of :class:`Unsqueeze`."""
        gymnasium.utils.RecordConstructorArgs.__init__(self)
        gymnasium.Wrapper.__init__(self, env)
        self.is_vector_env = getattr(env, 'is_vector_env', False)
        assert not self.is_vector_env, 'UnsqueezeWrapper does not support vectorized environments'

    def step(self, action):
        """The vector information will be unsqueezed to (1, dim) for agent training.

        Args:
            action: The action to take.

        Returns:
            The unsqueezed environment :meth:`step`
        """
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        obs, reward, cost, terminated, truncated = (
            np.expand_dims(x, axis=0) for x in (obs, reward, cost, terminated, truncated)
        )
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                info[k] = np.expand_dims(v, axis=0)

        return obs, reward, cost, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and returns a new observation.

        .. note::
            The vector information will be unsqueezed to (1, dim) for agent training.

        Args:
            seed (int or None, optional): Set the seed for the environment. Defaults to None.

        Returns:
            obs: The initial observation of the space.
            info: Some information logged by the environment.
        """
        obs, info = self.env.reset(**kwargs)
        obs = np.expand_dims(obs, axis=0)
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                info[k] = np.expand_dims(v, axis=0)

        return obs, info
