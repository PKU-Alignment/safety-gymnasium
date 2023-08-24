Basic Usage
===========

Installation
------------

.. code-block:: bash

    # From the Python Package Index (PyPI)
    pip install safety-gymnasium

    # From the source code
    git clone https://github.com/PKU-Alignment/safety-gymnasium.git
    cd safety-gymnasium
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
    '''
    Box([-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf   0.   0.
    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.],
    [inf inf inf inf inf inf inf inf inf inf inf inf  1.  1.  1.  1.  1.  1.
    1.  1.  1.  1.  1.  1.  1.  1.  1.  1.], (28,), float64)
    '''


    print(env.obs_space_dict)
    '''
    Dict('accelerometer': Box(-inf, inf, (3,), float64),
         'velocimeter': Box(-inf, inf, (3,), float64),
         'gyro': Box(-inf, inf, (3,), float64),
         'magnetometer': Box(-inf, inf, (3,), float64),
         'circle_lidar': Box(0.0, 1.0, (16,), float64))
    '''


    # position of each part in the obs is as same as it appears in the Dict above.
    print(obs)
    '''
	[0.         0.         9.81       0.         0.         0.
    0.         0.         0.         0.36647163 0.34014489 0.
    0.         0.         0.         0.         0.         0.
    0.         0.         0.         0.         0.         0.08518475
    0.93364224 0.84845749 0.         0.        ]
    '''


Action Space
------------

.. code-block:: python

    env = safety_gymnasium.make('SafetyPointCircle0-v0', render_mode='human')
    obs, info = env.reset()
    print(env.action_space)
    # Box(-1.0, 1.0, (2,), float64)

Render
------

We completely inherit the excellent API for render in Gymnasium.

.. Note::

    The set of supported modes varies per environment. (And some
    third-party environments may not support rendering at all.)
    By convention, if render_mode is:

    - **None (default)**: no render is computed.
    - **human**: render return None.
      The environment is continuously rendered in the current display or terminal. Usually for human consumption.
    - **rgb_array**: return a single frame representing the current state of the environment.
      A frame is a numpy.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
    - **rgb_array_list**: return a list of frames representing the states of the environment since the last reset.
      Each frame is a numpy.ndarray with shape (x, y, 3), as with `rgb_array`.
    - **depth_array**: return a single frame representing the current state of the environment.
      A frame is a numpy.ndarray with shape (x, y) representing depth values for an x-by-y pixel image.
    - **depth_array_list**: return a list of frames representing the states of the environment since the last reset.
      Each frame is a numpy.ndarray with shape (x, y), as with `depth_array`.

Debug with your keyboard
------------------------

.. Note::

    For simple agents, we offer the capability to control the robot's movement via the keyboard, facilitating debugging. Simply append a **Debug** suffix to the task name, such as **SafetyCarGoal2Debug-v0**, and utilize the keys `I`, `K`, `J`, and `L` to guide the robot's movement.

    For more intricate agents, you can also craft custom control logic based on specific peripherals. To achieve this, implement the `debug` method from the `BaseAgent` for the designated agent.
