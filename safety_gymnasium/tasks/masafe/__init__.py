from safety_gymnasium import register
from safety_gymnasium.tasks.masafe.coupled_half_cheetah import CoupledHalfCheetah
from safety_gymnasium.tasks.masafe.manyagent_ant import ManyAgentAntEnv
from safety_gymnasium.tasks.masafe.manyagent_swimmer import ManyAgentSwimmerEnv
from safety_gymnasium.tasks.masafe.mujoco_multi import MujocoMulti


register(
    id='2AgentAnt-v4',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'Ant-v4',
            'agent_conf': '2x4',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='2AgentAntDiag-v4',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'Ant-v4',
            'agent_conf': '2x4d',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='4AgentAnt-v4',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'Ant-v4',
            'agent_conf': '4x2',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='2AgentHalfCheetah-v4',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'HalfCheetah-v4',
            'agent_conf': '2x3',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='6AgentHalfCheetah-v4',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'HalfCheetah-v4',
            'agent_conf': '6x1',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='3AgentHopper-v4',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'Hopper-v4',
            'agent_conf': '3x1',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='2AgentHumanoid-v4',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'Humanoid-v4',
            'agent_conf': '9|8',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='2AgentHumanoidStandup-v4',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'HumanoidStandup-v4',
            'agent_conf': '9|8',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='2AgentReacher-v4',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'Reacher-v4',
            'agent_conf': '2x1',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='2AgentSwimmer-v4',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'Swimmer-v4',
            'agent_conf': '2x1',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='2AgentWalker2d-v4',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'Walker2d-v4',
            'agent_conf': '2x3',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='ManyAgentSwimmer-v0',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'manyagent_swimmer',
            'agent_conf': '10x2',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='ManyAgentAnt-v0',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'manyagent_ant',
            'agent_conf': '2x3',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)

register(
    id='CoupledHalfCheetah-v0',
    entry_point='safety_gymnasium.tasks.masafe.mujoco_multi:MujocoMulti',
    kwargs={
        'env_args': {
            'scenario': 'coupled_half_cheetah',
            'agent_conf': '1p1',
            'agent_obsk': 1,
            'episode_limit': 1000,
        },
    },
    max_episode_steps=1000,
)
