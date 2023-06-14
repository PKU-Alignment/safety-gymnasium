import time

import numpy as np

import safety_gymnasium


def main():
    # env_args = {"scenario": "coupled_half_cheetah",
    #               "agent_conf": "1p1",
    #               "agent_obsk": 1,
    #               "episode_limit": 1000}
    # env = MujocoMulti(env_args=env_args)
    # env_args = {"agent_conf": "2x4",
    #             "agent_obsk": 1,
    #             "episode_limit": 1000}
    env = safety_gymnasium.make('2AgentAnt-v4', render_mode='human')

    env_info = env.get_env_info()

    n_actions = env_info['n_actions']
    n_agents = env_info['n_agents']
    n_episodes = 10

    for e in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                np.nonzero(avail_actions)[0]
                action = np.random.uniform(-1.0, 1.0, n_actions)
                actions.append(action)

            obs, reward, cost, terminated, truncated, info = env.step(actions)
            info['state']
            episode_reward += reward

            time.sleep(0.1)
            env.render()

        print(f'Total reward in episode {e} = {episode_reward}')

    env.close()


if __name__ == '__main__':
    main()
