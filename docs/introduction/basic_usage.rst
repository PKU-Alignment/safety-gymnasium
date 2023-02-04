Basic Usage
===========

Installation
------------

.. code-block:: bash

    # From the Python Package Index (PyPI)
    pip install safety-gymnasium
    # From the source code
    git clone git@github.com:PKU-MARL/Safety-Gymnasium.git
    cd Safety-Gymnasium
    pip install -e .


Specification
-------------

`Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`__ provides a well-defined and widely accepted API by the RL Community, and our library exactly adheres to this specification and provides a Safe RL-specific interface.So researchers accustomed to Gymnasium can get started with our library at near zero migration cost, for some basic API and code tools refer to: `Gymnasium Documentation <https://www.gymlibrary.dev/>`__.

Initializing the environment
----------------------------

.. code-block:: python

    import safety_gymnasium
    env = safety_gymnasium.make('SafetyPointCircle0-v0', render_mode='human')
    '''
    Vision Environment
        env = safety_gymnasium.make('SafetyPointCircle0Vision-v0', render_mode='human')
    Keyboard Debug environment
    due to the complexity of the agent's inherent dynamics, only partial support for the agent.
	env = safety_gymnasium.make('SafetyPointCircle0Debug-v0', render_mode='human')
    '''
    obs, info = env.reset()
    # Set seeds
    # obs, _ = env.reset(seed=0)
    terminated, truncated = False, False
    ep_ret, ep_cost = 0, 0
    for _ in range(1000):
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        # modified for Safe RL, added cost
        obs, reward, cost, terminated, truncated, info = env.step(act)
        ep_ret += reward
        ep_cost += cost
        if terminated or truncated:
            observation, info = env.reset()

	env.close()

Observation Space
-----------------

.. code-block:: python

    env = safety_gymnasium.make('SafetyPointCircle0-v0', render_mode='human')
    obs, info = env.reset()
    print(env.observation_space)
    # Box(-inf, inf, (28,), float64)
    print(env.obs_space_dict)
    '''
    OrderedDict([
        ('accelerometer', Box(-inf, inf, (3,), float64)),
        ('velocimeter', Box(-inf, inf, (3,), float64)),
        ('gyro', Box(-inf, inf, (3,), float64)),
        ('magnetometer', Box(-inf, inf, (3,), float64)),
        ('circle_lidar', Box(0.0, 1.0, (16,), float64))
        ])
    '''
    # position of each part in the obs array
    print(sorted(env.obs_space_dict))
	# ['accelerometer', 'circle_lidar', 'gyro', 'magnetometer', 'velocimeter']


Action Space
------------

.. code-block:: python

    env = safety_gymnasium.make('SafetyPointCircle0-v0', render_mode='human')
    obs, info = env.reset()
    print(env.action_space)
    # Box(-1.0, 1.0, (2,), float64)
