# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json

from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.freight_franka_close_drawer import (
    FreightFrankaCloseDrawer,
    FreightFrankaPickAndPlace,
)
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.hand_base.multi_vec_task import (
    MultiVecTaskPython,
    SingleVecTaskPythonArm,
)
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.hand_base.vec_task import (
    VecTaskCPU,
    VecTaskGPU,
    VecTaskPython,
    VecTaskPythonArm,
)
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_finger import (
    ShadowHandCatchOver2Underarm_Safe_finger,
)
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_joint import (
    ShadowHandCatchOver2Underarm_Safe_joint,
)
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_finger import (
    ShadowHandOver_Safe_finger,
)
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_joint import (
    ShadowHandOver_Safe_joint,
)
from safety_gymnasium.tasks.safe_isaac_gym.utils.config import warn_task_name


def parse_task(args, cfg, cfg_train, sim_params, agent_index):
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    print('device_id', args.device_id)
    print('rl_device', args.rl_device)

    cfg['seed'] = cfg_train.get('seed', -1)
    cfg_task = cfg['env']
    cfg_task['seed'] = cfg['seed']

    if args.task_type == 'C++':
        if args.device == 'cpu':
            print('C++ CPU')
            task = rlgpu.create_task_cpu(args.task, json.dumps(cfg_task))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(device_id, -1, args.physics_engine, sim_params)
            else:
                task.init(device_id, device_id, args.physics_engine, sim_params)
            env = VecTaskCPU(
                task,
                rl_device,
                False,
                cfg_train.get('clip_observations', 5.0),
                cfg_train.get('clip_actions', 1.0),
            )
        else:
            print('C++ GPU')

            task = rlgpu.create_task_gpu(args.task, json.dumps(cfg_task))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(device_id, -1, args.physics_engine, sim_params)
            else:
                task.init(device_id, device_id, args.physics_engine, sim_params)
            env = VecTaskGPU(
                task,
                rl_device,
                cfg_train.get('clip_observations', 5.0),
                cfg_train.get('clip_actions', 1.0),
            )

    elif args.task_type == 'Python':
        print('Python')
        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                is_multi_agent=False,
            )
        except NameError as e:
            print(e)
            warn_task_name()
        if args.task == 'OneFrankaCabinet':
            env = VecTaskPythonArm(task, rl_device)
        else:
            env = VecTaskPython(task, rl_device)

    elif args.task_type == 'MultiAgent':
        print('MultiAgent')

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=True,
            )
        except NameError as e:
            print(e)
            warn_task_name()
        env = MultiVecTaskPython(task, rl_device)
    return task, env
