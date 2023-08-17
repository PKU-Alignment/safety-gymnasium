
Safety-Gymnasium
================

Safety-Gymnasium is a standard API for safe reinforcement learning, and a diverse collection of reference environments.

.. image:: _static/images/car_demo.gif
   :alt: racecar
   :width: 500
   :align: center

.. code-block:: python

   import safety_gymnasium

   env = safety_gymnasium.vector.make("SafetyCarGoal1-v0", render_mode="human", num_envs=8)
   observation, info = env.reset(seed=0)

   for _ in range(1000):
      action = env.action_space.sample()  # this is where you would insert your policy
      observation, reward, cost, terminated, truncated, info = env.step(action)

      if terminated or truncated:
         observation, info = env.reset()

   env.close()


.. toctree::
   :hidden:
   :caption: INTRODUCTION

   introduction/about_safety_gymnasium
   introduction/basic_usage


.. toctree::
   :hidden:
   :caption: COMPONENTS OF ENVIRONMENTS

   components_of_environments/agents
   components_of_environments/objects
   components_of_environments/tasks


.. toctree::
   :hidden:
   :caption: ENVIRONMENTS

   environments/safe_navigation
   environments/safe_velocity
   environments/safe_vision
   environments/safe_multi_agent
   environments/safe_isaac_gym


.. toctree::
   :hidden:
   :caption: API

   api/bases
   api/builder
   api/utils


.. toctree::
   :hidden:
   :caption: Development

`Github <https://github.com/PKU-Alignment/safety-gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__
