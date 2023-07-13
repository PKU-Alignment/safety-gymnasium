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
"""Wrapper for limiting the time steps of an environment."""


from gymnasium.wrappers.autoreset import AutoResetWrapper


class SafeAutoResetWrapper(AutoResetWrapper):
    """A class for providing an automatic reset functionality for gymnasium environments when calling :meth:`step`.

     - ``new_obs`` is the first observation after calling ``self.env.reset()``
     - ``final_reward`` is the reward after calling ``self.env.step()``, prior to calling ``self.env.reset()``.
     - ``final_terminated`` is the terminated value before calling ``self.env.reset()``.
     - ``final_truncated`` is the truncated value before calling ``self.env.reset()``. Both ``final_terminated`` and ``final_truncated`` cannot be False.
     - ``info`` is a dict containing all the keys from the info dict returned by the call to ``self.env.reset()``,
       with an additional key "final_observation" containing the observation returned by the last call to ``self.env.step()``
       and "final_info" containing the info dict returned by the last call to ``self.env.step()``.

    Warning: When using this wrapper to collect roll-outs, note that when :meth:`Env.step` returns ``terminated`` or ``truncated``, a
        new observation from after calling :meth:`Env.reset` is returned by :meth:`Env.step` alongside the
        final reward, terminated and truncated state from the previous episode.
        If you need the final state from the previous episode, you need to retrieve it via the
        "final_observation" key in the info dict.
        Make sure you know what you're doing if you use this wrapper!
    """  # pylint: disable=line-too-long

    def step(self, action):
        """A class for providing an automatic reset functionality for gymnasium environments when calling :meth:`step`.

        Args:
            env (gym.Env): The environment to apply the wrapper
        """  # pylint: disable=line-too-long
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            new_obs, new_info = self.env.reset()
            assert (
                'final_observation' not in new_info
            ), 'info dict cannot contain key "final_observation" '
            assert 'final_info' not in new_info, 'info dict cannot contain key "final_info" '

            new_info['final_observation'] = obs
            new_info['final_info'] = info

            obs = new_obs
            info = new_info

        return obs, reward, cost, terminated, truncated, info
