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
"""Wrapper for rescaling actions to within a max and min action."""

from __future__ import annotations

import gymnasium
import numpy as np


class SafeRescaleAction(gymnasium.ActionWrapper, gymnasium.utils.RecordConstructorArgs):
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

    The base environment :attr:`env` must have an action space of type :class:`spaces.Box`. If :attr:`min_action`
    or :attr:`max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    Example:
        >>> import safety_gymnasium
        >>> from safety_gymnasium.wrappers import RescaleAction
        >>> import numpy as np
        >>> env = safety_gymnasium.make("SafetyPointGoal1-v0")
        >>> env = RescaleAction(env, min_action=-1, max_action=1)

    """

    def __init__(
        self,
        env: gymnasium.Env,
        min_action: float | np.ndarray,
        max_action: float | np.ndarray,
    ) -> None:
        """Initializes the :class:`RescaleAction` wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_action (float, int or np.ndarray): The min values for each action.
                This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action.
                This may be a numpy array or a scalar.
        """
        gymnasium.utils.RecordConstructorArgs.__init__(
            self,
            min_action=min_action,
            max_action=max_action,
        )
        gymnasium.ActionWrapper.__init__(self, env)

        self.min_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + min_action
        )
        self.max_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + max_action
        )

    def action(self, action):
        """Rescales the action

        Rescales the action affinely from [:attr:`min_action`, :attr:`max_action`] to the action
        space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        low = self.env.action_space.low
        high = self.env.action_space.high
        return low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
