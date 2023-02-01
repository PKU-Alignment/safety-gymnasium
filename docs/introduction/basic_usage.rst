基本使用
========

安装
-----

.. code-block:: bash

    # 通过pypi
    pip install Safety-Gymnasium
    # 通过github/源码
    git clone https://github.com/PKU-MARL/safety-gymnasium
    cd safety-gymnasium
    pip install -e .


规范
-----

`Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`__ 提供了良好定义且被RL Community广泛接受的API规范，我们的库完全遵循该规范，并且提供了方便Safe RL研究所必要的接口。因此习惯于Gymnasium的研究者可近乎0迁移成本地上手我们的库，关于一些基础API和代码工具可参考：
`Gymnasium文档 <https://www.gymlibrary.dev/>`__ 。

初始化环境
----------

.. code-block:: python

    env = safety_gymnasium.make('SafetyPointCircle0-v0', render_mode='human')
    # Vision环境
    # env = safety_gymnasium.make('SafetyPointCircle0Vision-v0', render_mode='human')
    # 键盘Debug环境（由于agent本身的动力学复杂性，仅支持部分agent。）
	# env = safety_gymnasium.make('SafetyPointCircle0Debug-v0', render_mode='human')
    obs, info = env.reset()
    # 设置seed：
    # obs, _ = env.reset(seed=0)
    terminated, truncated = False, False
    ep_ret, ep_cost = 0, 0
    for _ in range(1000):
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        # 针对Safe RL做了修改，增加了cost返回值
        obs, reward, cost, terminated, truncated, info = env.step(act)
        ep_ret += reward
        ep_cost += cost
        if terminated or truncated:
            observation, info = env.reset()

	env.close()

状态空间
--------

.. code-block:: python

    env = safety_gymnasium.make('SafetyPointCircle0-v0', render_mode='human')
    obs, info = env.reset()
    print(env.observation_space)
    # Box(-inf, inf, (28,), float64)
    print(env.obs_space_dict)
    # OrderedDict([('accelerometer', Box(-inf, inf, (3,), float64)), ('velocimeter', Box(-inf, inf, (3,), float64)), ('gyro', Box(-inf, inf, (3,), float64)), ('magnetometer', Box(-inf, inf, (3,), float64)), ('circle_lidar', Box(0.0, 1.0, (16,), float64))])
    # 每一个部分在obs数组当中的位置：
    print(sorted(env.obs_space_dict))
	# ['accelerometer', 'circle_lidar', 'gyro', 'magnetometer', 'velocimeter']


动作空间
---------

.. code-block:: python

    env = safety_gymnasium.make('SafetyPointCircle0-v0', render_mode='human')
    obs, info = env.reset()
    print(env.action_space)
    # Box(-1.0, 1.0, (2,), float64)


