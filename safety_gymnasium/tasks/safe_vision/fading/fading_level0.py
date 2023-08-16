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
"""Fading level 0."""

import mujoco

from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0


class FadingEasyLevel0(GoalLevel0):
    """An agent must navigate to a goal.

    The goal will gradually disappear over time.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.fadding_steps = 150
        self.fadding_objects = [self.goal]  # pylint: disable=no-member
        self.objects_map_ids: dict

    def _init_fading_ids(self):
        self.objects_map_ids = {}

        for obj in self.fadding_objects:
            if obj.name not in self.objects_map_ids:
                self.objects_map_ids[obj.name] = []
            if not hasattr(obj, 'num'):
                self.objects_map_ids[obj.name].append(
                    mujoco.mj_name2id(  # pylint: disable=no-member
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,  # pylint: disable=no-member
                        obj.name,
                    ),
                )
            else:
                for i in range(obj.num):
                    self.objects_map_ids[obj.name].append(
                        mujoco.mj_name2id(  # pylint: disable=no-member
                            self.model,
                            mujoco.mjtObj.mjOBJ_GEOM,  # pylint: disable=no-member
                            obj.name[:-1] + str(i),
                        ),
                    )

    def set_objects_alpha(self, object_name, alpha):
        """Set the alpha value of the object via ids of them in MuJoCo."""
        for i in self.objects_map_ids[object_name]:
            self.model.geom_rgba[i][-1] = alpha

    def linear_decrease_alpha(self, object_name, alpha):
        """Linearly decrease the alpha value of the object via ids of them in MuJoCo."""
        for i in self.objects_map_ids[object_name]:
            self.model.geom_rgba[i][-1] = max(
                self.model.geom_rgba[i][-1] - alpha / self.fadding_steps,
                0,
            )

    def specific_reset(self):
        if not hasattr(self, 'objects_map_ids'):
            self._init_fading_ids()
        for obj in self.fadding_objects:
            self.set_objects_alpha(obj.name, getattr(self, obj.name).alpha)

    def specific_step(self):
        for obj in self.fadding_objects:
            self.linear_decrease_alpha(obj.name, getattr(self, obj.name).alpha)

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        # pylint: disable=no-member
        goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.goal.name)
        self.model.geom_rgba[goal_id][-1] = self.goal.alpha
        self.last_dist_goal = self.dist_goal()
        # pylint: enable=no-member


class FadingHardLevel0(FadingEasyLevel0):
    """The goal will disappear more quickly."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.fadding_steps = 75
