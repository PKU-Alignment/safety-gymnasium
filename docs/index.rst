
Safety Gymnasium is a standard API for safe reinforcement learning, and a diverse collection of reference environments
======================================================================================================================

.. image:: _static/images/racecar_demo.gif
   :alt: racecar    
   :width: 200
   :align: center

.. code-block:: python

   import safety_gymnasium
   env = safety_gymnasium.make("Safety_RacecarGoal1-v0", render_mode="human")
   observation, info = env.reset(seed=42)
   for _ in range(1000):
      action = env.action_space.sample()  # this is where you would insert your policy
      observation, reward, terminated, truncated, info = env.step(action)

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
   :caption: COMPONETS OF ENVIRONMENTS

   components_of_environments/agents
   components_of_environments/objects
   components_of_environments/tasks


.. toctree::
   :hidden:
   :caption: ENVIRONMENTS

   environments/safe_navigation


.. toctree::
   :hidden:
   :caption: API

   api/bases
   api/builder
   api/utils


.. toctree::
   :hidden:
   :caption: Development

`Github <https://github.com/PKU-MARL/Safety-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-MARL/Safety-Gymnasium/blob/main/CONTRIBUTING.md>`__
