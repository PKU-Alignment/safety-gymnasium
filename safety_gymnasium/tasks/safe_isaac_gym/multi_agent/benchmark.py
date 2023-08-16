import argparse
import shlex
import subprocess

from safepo.utils.config import multi_agent_velocity_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=list(multi_agent_velocity_map.keys()),
        help='the ids of the environment to benchmark',
    )
    parser.add_argument(
        '--algo',
        nargs='+',
        default=[
            'mappo',
            'mappolag',
            'macpo',
            'happo',
        ],
        help='the ids of the algorithm to benchmark',
    )
    parser.add_argument('--num-seeds', type=int, default=3, help='the number of random seeds')
    parser.add_argument('--start-seed', type=int, default=0, help='the number of the starting seed')
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='the number of workers to run benchmark experimenets',
    )
    parser.add_argument(
        '--experiment', type=str, default='benchmark_multi_env', help='name of the experiment'
    )
    args = parser.parse_args()

    return args


def run_experiment(command: str):
    command_list = shlex.split(command)
    print(f'running {command}')
    fd = subprocess.Popen(command_list)
    return_code = fd.wait()
    assert return_code == 0


if __name__ == '__main__':
    args = parse_args()

    commands = []

    for seed in range(0, args.num_seeds):
        for task in args.tasks:
            for algo in args.algo:
                agen_conf = multi_agent_velocity_map[task]['agent_conf']
                scenario = multi_agent_velocity_map[task]['scenario']
                commands += [
                    ' '.join(
                        [
                            f'python {algo}.py',
                            '--agent-conf',
                            agen_conf,
                            '--scenario',
                            scenario,
                            '--seed',
                            str(args.start_seed + 1000 * seed),
                            '--write-terminal',
                            'False',
                            '--experiment',
                            args.experiment,
                            '--total-steps',
                            '2000',
                            '--num-envs',
                            '1',
                        ]
                    )
                ]

    print('======= commands to run:')
    for command in commands:
        print(command)

    if args.workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(
            max_workers=args.workers, thread_name_prefix='safepo-benchmark-worker-'
        )
        for command in commands:
            executor.submit(run_experiment, command)
        executor.shutdown(wait=True)
    else:
        print(
            'not running the experiments because --workers is set to 0; just printing the commands to run'
        )
