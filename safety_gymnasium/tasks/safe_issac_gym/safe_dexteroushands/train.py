# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from algorithms import REGISTRY
from algorithms.module import Actor, Critic

from utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
from utils.Logger import EpochLogger
from utils.parse_task import parse_task
from utils.process_sarl import *


def train(logdir):
    print('Algorithm: ', args.algo)
    logger = EpochLogger(args.algo, args.task, args.seed)
    agent_index = get_AgentIndex(cfg)
    if args.algo in ['ppol', 'focops', 'p3o', 'pcpo', 'cpo', 'trpol', 'ppo']:
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
        learn_cfg = cfg_train['learn']
        is_testing = learn_cfg['test']
        # is_testing = True
        # Override resume and testing flags if they are passed as parameters.
        if args.model_dir != '':
            is_testing = True
            chkpt_path = args.model_dir

        logdir = logdir + '_seed{}'.format(env.task.cfg['seed'])
        iterations = cfg_train['learn']['max_iterations']
        if args.max_iterations > 0:
            iterations = args.max_iterations

        """Set up the agent system for training or inferencing."""
        agent = REGISTRY[args.algo](
            vec_env=env,
            logger=logger,
            actor_class=Actor,
            critic_class=Critic,
            cost_critic_class=Critic,
            cost_lim=args.cost_lim,
            num_transitions_per_env=learn_cfg['nsteps'],
            num_learning_epochs=learn_cfg['noptepochs'],
            num_mini_batches=learn_cfg['nminibatches'],
            clip_param=learn_cfg['cliprange'],
            gamma=learn_cfg['gamma'],
            lam=learn_cfg['lam'],
            init_noise_std=learn_cfg.get('init_noise_std', 0.3),
            value_loss_coef=learn_cfg.get('value_loss_coef', 2.0),
            entropy_coef=learn_cfg['ent_coef'],
            learning_rate=learn_cfg['optim_stepsize'],
            max_grad_norm=learn_cfg.get('max_grad_norm', 2.0),
            use_clipped_value_loss=learn_cfg.get('use_clipped_value_loss', False),
            schedule=learn_cfg.get('schedule', 'fixed'),
            desired_kl=learn_cfg.get('desired_kl', None),
            model_cfg=cfg_train['policy'],
            device=env.rl_device,
            sampler=learn_cfg.get('sampler', 'sequential'),
            log_dir=logdir,
            is_testing=is_testing,
            print_log=learn_cfg['print_log'],
            apply_reset=False,
            asymmetric=(env.num_states > 0),
        )
        if is_testing and args.model_dir != '':
            print(f'Loading model from {chkpt_path}')
            agent.test(chkpt_path)
        elif args.model_dir != '':
            print(f'Loading model from {chkpt_path}')
            agent.load(chkpt_path)
        agent.run(
            num_learning_iterations=iterations,
            log_interval=cfg_train['learn']['save_interval'],
        )

    else:
        print(
            'Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo,maddpg,sac,td3,trpo,ppo,ddpg]',
        )


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)

    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get('seed', -1), cfg_train.get('torch_deterministic', False))
    train(logdir)
