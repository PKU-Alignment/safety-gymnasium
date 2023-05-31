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
"""Env wrappers."""

from typing import Callable

import gymnasium

from safety_gymnasium.wrappers.autoreset import SafeAutoResetWrapper
from safety_gymnasium.wrappers.env_checker import SafePassiveEnvChecker
from safety_gymnasium.wrappers.gymnasium_conversion import (
    Gymnasium2SafetyGymnasium,
    SafetyGymnasium2Gymnasium,
)
from safety_gymnasium.wrappers.normalize import (
    SafeNormalizeCost,
    SafeNormalizeObservation,
    SafeNormalizeReward,
)
from safety_gymnasium.wrappers.rescale_action import SafeRescaleAction
from safety_gymnasium.wrappers.time_limit import SafeTimeLimit
from safety_gymnasium.wrappers.unsqueeze import SafeUnsqueeze


__all__ = [
    'SafeAutoResetWrapper',
    'SafePassiveEnvChecker',
    'SafeTimeLimit',
    'SafetyGymnasium2Gymnasium',
    'Gymnasium2SafetyGymnasium',
    'with_gymnasium_wrappers',
    'SafeNormalizeObservation',
    'SafeNormalizeCost',
    'SafeNormalizeReward',
    'SafeUnsqueeze',
    'SafeRescaleAction',
]


def with_gymnasium_wrappers(
    env: gymnasium.Env,
    *wrappers: Callable[[gymnasium.Env], gymnasium.Env],
) -> gymnasium.Env:
    """Wrap an environment with Gymnasium wrappers.

    Example::

        >>> env = safety_gymnasium.wrappers.with_gymnasium_wrappers(
        ...     safety_gymnasium.make(env_id),
        ...     gymnasium.wrappers.SomeWrapper1,
        ...     functools.partial(gymnasium.wrappers.SomeWrapper2, argname1=arg1, argname2=arg2),
        ...     ...,
        ...     gymnasium.wrappers.SomeWrapperN,
        ... )
    """
    for wrapper in (SafetyGymnasium2Gymnasium, *wrappers, Gymnasium2SafetyGymnasium):
        if not callable(wrapper):  # wrapper class or a partial function
            raise TypeError(f'Wrapper {wrapper} is not callable.')
        env = wrapper(env)
    return env
