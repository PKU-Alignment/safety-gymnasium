Safe Velocity
================

Safe velocity tasks 基于 `Gymnasium's MuJoCo <https://gymnasium.farama.org/environments/mujoco/>`__ 系列智能体引入了速度约束，对于agent特定的信息，请查阅Gymnasium的文档.

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

Velocity tasks are also an important class of tasks that apply RL to reality, requiring an agent to move as quickly as possible while adhering to **velocity constraint**. These tasks have significant implications in various domains, including robotics, autonomous vehicles, and industrial automation.

Costs
-----

If **velocity of current step** exceeds the **threshold of velocity**, then receive an scalar signal 1, otherwise 0.

We can formulate it as follow:

.. math:: cost=bool(V_{current} > V_{threshold})

我们进行了大量实验。The velocity threshold is set to **50%** of the agent's maximum velocity achieved after the convergence of the **Proximal Policy Optimization (PPO)** algorithm trained via **1e6 steps**.

.. Note::
    对于Swimmer，我们只使用了其在X轴上的速度来设定约束，这是因为它的运动依赖于它的摆动，这将在Y轴上产生速度。
    其余智能体是使用其在X-Y轴上所有可能的速度的矢量和来设定约束的，一个粗略的表达如下：

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
| SafetyHopperVelocity-v1      | 0.37315            |
+------------------------------+--------------------+
| SafetyAntVelocity-v1         | 2.5745             |
+------------------------------+--------------------+
| SafetyHumanoidVelocity-v1    | 2.3475             |
+------------------------------+--------------------+
| SafetyWalker2dVelocity-v1    | 1.7075             |
+------------------------------+--------------------+
| SafetyHalfCheetahVelocity-v1 | 2.8795             |
+------------------------------+--------------------+
| SafetySwimmerVelocity-v1     | 0.04845            |
+------------------------------+--------------------+
