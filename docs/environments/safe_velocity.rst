Safe Velocity
================

.. _SingleVelocity:

The Safe velocity tasks introduce velocity constraints for agents based on the `Gymnasium's MuJoCo-v4 <https://gymnasium.farama.org/environments/mujoco/>`__ series. For specific information regarding the agents, please refer to the Gymnasium documentation.

Velocity tasks are also an important class of tasks that apply RL to reality, requiring an agent to move as quickly as possible while adhering to **velocity constraint**. These tasks have significant implications in various domains, including robotics, autonomous vehicles, and industrial automation.

+-----------------------------+------------------------------------------------------------------+
| **Import**                  | ``safety_gymnasium.make("Safety[Agent]Velocity-v1")``            |
+-----------------------------+------------------------------------------------------------------+

.. list-table::

    * - .. figure:: ../_static/images/hopper_vel_unsafe.gif
            :width: 300px
        .. centered:: HopperVelocity UNSAFE
      - .. figure:: ../_static/images/hopper_vel_safe.gif
            :width: 300px
        .. centered:: HopperVelocity SAFE
    * - .. figure:: ../_static/images/ant_vel_unsafe.gif
            :width: 300px
        .. centered:: AntVelocity UNSAFE
      - .. figure:: ../_static/images/ant_vel_safe.gif
            :width: 300px
        .. centered:: AntVelocity SAFE
    * - .. figure:: ../_static/images/humanoid_vel_unsafe.gif
            :width: 300px
        .. centered:: HumanoidVelocity UNSAFE
      - .. figure:: ../_static/images/humanoid_vel_safe.gif
            :width: 300px
        .. centered:: HumanoidVelocity SAFE
    * - .. figure:: ../_static/images/walker2d_vel_unsafe.gif
            :width: 300px
        .. centered:: Walker2dVelocity UNSAFE
      - .. figure:: ../_static/images/walker2d_vel_safe.gif
            :width: 300px
        .. centered:: Walker2dVelocity SAFE
    * - .. figure:: ../_static/images/half_cheetah_vel_unsafe.gif
            :width: 300px
        .. centered:: HalfCheetahVelocity UNSAFE
      - .. figure:: ../_static/images/half_cheetah_vel_safe.gif
            :width: 300px
        .. centered:: HalfCheetahVelocity SAFE
    * - .. figure:: ../_static/images/swimmer_vel_unsafe.gif
            :width: 300px
        .. centered:: SwimmerVelocity UNSAFE
      - .. figure:: ../_static/images/swimmer_vel_safe.gif
            :width: 300px
        .. centered:: SwimmerVelocity SAFE

Costs
-----

If **velocity of current step** exceeds the **threshold of velocity**, then receive an scalar signal 1, otherwise 0.

We can formulate it as follow:

.. math:: cost=bool(V_{current} > V_{threshold})

After conducting extensive experiments. The velocity threshold is set to **50%** of the agent's maximum velocity achieved after the convergence of the **Proximal Policy Optimization (PPO)** algorithm trained via **1e7 steps**.

.. Note::
    For the **Swimmer**, we only set constraints based on its velocity in the X-axis, as its movement relies on its oscillation, which generates velocity in the Y-axis.

    For the remaining agents, constraints were set based on the vector sum of all possible velocities in the X-Y plane. A **concise representation** is as follows:

    .. code-block:: python

        if 'y_velocity' not in agent_infomation:
            agent_velocity = np.abs(agent_infomation['x_velocity'])
        else:
            agent_velocity = np.sqrt(agent_infomation['x_velocity'] ** 2 + agent_infomation['y_velocity'] ** 2)

+------------------------------+--------------------+
| Environment                  | Velocity Threshold |
+==============================+====================+
| SafetyHopperVelocity-v1      | 0.7402             |
+------------------------------+--------------------+
| SafetyAntVelocity-v1         | 2.6222             |
+------------------------------+--------------------+
| SafetyHumanoidVelocity-v1    | 1.4149             |
+------------------------------+--------------------+
| SafetyWalker2dVelocity-v1    | 2.3415             |
+------------------------------+--------------------+
| SafetyHalfCheetahVelocity-v1 | 3.2096             |
+------------------------------+--------------------+
| SafetySwimmerVelocity-v1     | 0.2282             |
+------------------------------+--------------------+

Version History
---------------

v0:

+------------------------------+--------------------+
| Environment                  | Velocity Threshold |
+==============================+====================+
| SafetyHopperVelocity-v0      | 0.37315            |
+------------------------------+--------------------+
| SafetyAntVelocity-v0         | 2.5745             |
+------------------------------+--------------------+
| SafetyHumanoidVelocity-v0    | 2.3475             |
+------------------------------+--------------------+
| SafetyWalker2dVelocity-v0    | 1.7075             |
+------------------------------+--------------------+
| SafetyHalfCheetahVelocity-v0 | 2.8795             |
+------------------------------+--------------------+
| SafetySwimmerVelocity-v0     | 0.04845            |
+------------------------------+--------------------+
