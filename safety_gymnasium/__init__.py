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
"""Safety-Gymnasium Environments."""

import copy

from gymnasium import make as gymnasium_make
from gymnasium import register as gymnasium_register

from safety_gymnasium import vector, wrappers
from safety_gymnasium.tasks.safe_multi_agent.tasks.velocity.safe_mujoco_multi import make_ma
from safety_gymnasium.utils.registration import make, register
from safety_gymnasium.version import __version__


__all__ = [
    'register',
    'make',
    'gymnasium_make',
    'gymnasium_register',
]

VERSION = 'v0'
ROBOT_NAMES = ('Point', 'Car', 'Doggo', 'Racecar', 'Ant')
MAKE_VISION_ENVIRONMENTS = True
MAKE_DEBUG_ENVIRONMENTS = True

# ========================================
# Helper Methods for Easy Registration
# ========================================

PREFIX = 'Safety'

robots = ROBOT_NAMES


def __register_helper(env_id, entry_point, spec_kwargs=None, **kwargs):
    """Register a environment to both Safety-Gymnasium and Gymnasium registry."""
    env_name, dash, version = env_id.partition('-')
    if spec_kwargs is None:
        spec_kwargs = {}

    register(
        id=env_id,
        entry_point=entry_point,
        kwargs=spec_kwargs,
        **kwargs,
    )
    gymnasium_register(
        id=f'{env_name}Gymnasium{dash}{version}',
        entry_point='safety_gymnasium.wrappers.gymnasium_conversion:make_gymnasium_environment',
        kwargs={'env_id': f'{env_name}Gymnasium{dash}{version}', **copy.deepcopy(spec_kwargs)},
        **kwargs,
    )


def __combine(tasks, agents, max_episode_steps):
    """Combine tasks and agents together to register environment tasks."""
    for task_name, task_config in tasks.items():
        # Vector inputs
        for robot_name in agents:
            env_id = f'{PREFIX}{robot_name}{task_name}-{VERSION}'
            combined_config = copy.deepcopy(task_config)
            combined_config.update({'agent_name': robot_name})

            __register_helper(
                env_id=env_id,
                entry_point='safety_gymnasium.builder:Builder',
                spec_kwargs={'config': combined_config, 'task_id': env_id},
                max_episode_steps=max_episode_steps,
            )

            if MAKE_VISION_ENVIRONMENTS:
                # Vision inputs
                vision_env_name = f'{PREFIX}{robot_name}{task_name}Vision-{VERSION}'
                vision_config = {
                    'observe_vision': True,
                    'observation_flatten': False,
                }
                vision_config.update(combined_config)
                __register_helper(
                    env_id=vision_env_name,
                    entry_point='safety_gymnasium.builder:Builder',
                    spec_kwargs={'config': vision_config, 'task_id': env_id},
                    max_episode_steps=max_episode_steps,
                )

            if MAKE_DEBUG_ENVIRONMENTS and robot_name in ['Point', 'Car', 'Racecar']:
                # Keyboard inputs for debugging
                debug_env_name = f'{PREFIX}{robot_name}{task_name}Debug-{VERSION}'
                debug_config = {'debug': True}
                debug_config.update(combined_config)
                __register_helper(
                    env_id=debug_env_name,
                    entry_point='safety_gymnasium.builder:Builder',
                    spec_kwargs={'config': debug_config, 'task_id': env_id},
                    max_episode_steps=max_episode_steps,
                )


# ----------------------------------------
# Safety Navigation
# ----------------------------------------

# Button Environments
# ----------------------------------------
button_tasks = {'Button0': {}, 'Button1': {}, 'Button2': {}}
__combine(button_tasks, robots, max_episode_steps=1000)


# Push Environments
# ----------------------------------------
push_tasks = {'Push0': {}, 'Push1': {}, 'Push2': {}}
__combine(push_tasks, robots, max_episode_steps=1000)


# Goal Environments
# ----------------------------------------
goal_tasks = {'Goal0': {}, 'Goal1': {}, 'Goal2': {}}
__combine(goal_tasks, robots, max_episode_steps=1000)


# Circle Environments
# ----------------------------------------
circle_tasks = {'Circle0': {}, 'Circle1': {}, 'Circle2': {}}
__combine(circle_tasks, robots, max_episode_steps=500)


# Run Environments
# ----------------------------------------
run_tasks = {'Run0': {}}
__combine(run_tasks, robots, max_episode_steps=500)


# ----------------------------------------
# Safety Vision
# ----------------------------------------

# Building Button Environments
# ----------------------------------------
building_button_tasks = {
    'BuildingButton0': {'floor_conf.type': 'mud'},
    'BuildingButton1': {'floor_conf.type': 'mud'},
    'BuildingButton2': {'floor_conf.type': 'mud'},
}
__combine(building_button_tasks, robots, max_episode_steps=1000)


# Building Push Environments
# ----------------------------------------
building_push_tasks = {
    'BuildingPush0': {'floor_conf.type': 'mud'},
    'BuildingPush1': {'floor_conf.type': 'mud'},
    'BuildingPush2': {'floor_conf.type': 'mud'},
}
__combine(building_push_tasks, robots, max_episode_steps=1000)


# Building Goal Environments
# ----------------------------------------
building_goal_tasks = {
    'BuildingGoal0': {'floor_conf.type': 'mud'},
    'BuildingGoal1': {'floor_conf.type': 'mud'},
    'BuildingGoal2': {'floor_conf.type': 'mud'},
}
__combine(building_goal_tasks, robots, max_episode_steps=1000)


# Race Environments
# ----------------------------------------
race_tasks = {
    'Race0': {'floor_conf.type': 'village'},
    'Race1': {'floor_conf.type': 'village'},
    'Race2': {'floor_conf.type': 'village'},
}
__combine(race_tasks, robots, max_episode_steps=500)


# Racing Environments
# ----------------------------------------
race_tasks = {
    'FormulaOne0': {'floor_conf.type': 'none'},
    'FormulaOne1': {'floor_conf.type': 'none'},
    'FormulaOne2': {'floor_conf.type': 'none'},
}
__combine(race_tasks, robots, max_episode_steps=50000000000000)


# Fading Environments
# ----------------------------------------
fading_tasks = {'FadingEasy0': {}, 'FadingEasy1': {}, 'FadingEasy2': {}}
__combine(fading_tasks, robots, max_episode_steps=1000)

fading_tasks = {'FadingHard0': {}, 'FadingHard1': {}, 'FadingHard2': {}}
__combine(fading_tasks, robots, max_episode_steps=1000)


# ----------------------------------------
# Safety Velocity
# ----------------------------------------

__register_helper(
    env_id='SafetyHalfCheetahVelocity-v0',
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_half_cheetah_velocity_v0:SafetyHalfCheetahVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

__register_helper(
    env_id='SafetyHopperVelocity-v0',
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_hopper_velocity_v0:SafetyHopperVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

__register_helper(
    env_id='SafetySwimmerVelocity-v0',
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_swimmer_velocity_v0:SafetySwimmerVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

__register_helper(
    env_id='SafetyWalker2dVelocity-v0',
    max_episode_steps=1000,
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_walker2d_velocity_v0:SafetyWalker2dVelocityEnv',
)

__register_helper(
    env_id='SafetyAntVelocity-v0',
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_ant_velocity_v0:SafetyAntVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

__register_helper(
    env_id='SafetyHumanoidVelocity-v0',
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_humanoid_velocity_v0:SafetyHumanoidVelocityEnv',
    max_episode_steps=1000,
)

__register_helper(
    env_id='SafetyHalfCheetahVelocity-v1',
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_half_cheetah_velocity_v1:SafetyHalfCheetahVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

__register_helper(
    env_id='SafetyHopperVelocity-v1',
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_hopper_velocity_v1:SafetyHopperVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

__register_helper(
    env_id='SafetySwimmerVelocity-v1',
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_swimmer_velocity_v1:SafetySwimmerVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

__register_helper(
    env_id='SafetyWalker2dVelocity-v1',
    max_episode_steps=1000,
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_walker2d_velocity_v1:SafetyWalker2dVelocityEnv',
)

__register_helper(
    env_id='SafetyAntVelocity-v1',
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_ant_velocity_v1:SafetyAntVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

__register_helper(
    env_id='SafetyHumanoidVelocity-v1',
    entry_point='safety_gymnasium.tasks.safe_velocity.safety_humanoid_velocity_v1:SafetyHumanoidVelocityEnv',
    max_episode_steps=1000,
)


def __combine_multi(tasks, agents, max_episode_steps):
    """Combine tasks and agents together to register environment tasks."""
    for task_name, task_config in tasks.items():
        # Vector inputs
        for robot_name in agents:
            env_id = f'{PREFIX}{robot_name}{task_name}-{VERSION}'
            combined_config = copy.deepcopy(task_config)
            combined_config.update({'agent_name': robot_name})

            __register_helper(
                env_id=env_id,
                entry_point='safety_gymnasium.tasks.safe_multi_agent.builder:Builder',
                spec_kwargs={'config': combined_config, 'task_id': env_id},
                max_episode_steps=max_episode_steps,
                disable_env_checker=True,
            )

            if MAKE_VISION_ENVIRONMENTS:
                # Vision inputs
                vision_env_name = f'{PREFIX}{robot_name}{task_name}Vision-{VERSION}'
                vision_config = {
                    'observe_vision': True,
                    'observation_flatten': False,
                }
                vision_config.update(combined_config)
                __register_helper(
                    env_id=vision_env_name,
                    entry_point='safety_gymnasium.tasks.safe_multi_agent.builder:Builder',
                    spec_kwargs={'config': vision_config, 'task_id': env_id},
                    max_episode_steps=max_episode_steps,
                    disable_env_checker=True,
                )

            if MAKE_DEBUG_ENVIRONMENTS and robot_name in ['Point', 'Car', 'Racecar']:
                # Keyboard inputs for debugging
                debug_env_name = f'{PREFIX}{robot_name}{task_name}Debug-{VERSION}'
                debug_config = {'debug': True}
                debug_config.update(combined_config)
                __register_helper(
                    env_id=debug_env_name,
                    entry_point='safety_gymnasium.tasks.safe_multi_agent.builder:Builder',
                    spec_kwargs={'config': debug_config, 'task_id': env_id},
                    max_episode_steps=max_episode_steps,
                    disable_env_checker=True,
                )


# ----------------------------------------
# Safety Multi-Agent
# ----------------------------------------

# Multi Goal Environments
# ----------------------------------------
multi_goal_tasks = {'MultiGoal0': {}, 'MultiGoal1': {}, 'MultiGoal2': {}}
__combine_multi(multi_goal_tasks, robots, max_episode_steps=1000)
