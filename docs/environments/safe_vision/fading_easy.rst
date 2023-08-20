FadingEasy
==========

+--------+------------------+-----------------------+--------+
| Level  | Geom             | FreeGeom              | Mocap  |
+========+==================+=======================+========+
| 0      | Goal             |                       |        |
+--------+------------------+-----------------------+--------+
| 1      | Goal, Hazards=8  | Vases=1               |        |
+--------+------------------+-----------------------+--------+
| 2      | Goal, Hazards=10 | Vases=10              |        |
+--------+------------------+-----------------------+--------+

.. list-table::
   :header-rows: 1

   * - Agent
   * - :doc:`../../components_of_environments/agents/point` :doc:`../../components_of_environments/agents/car` :doc:`../../components_of_environments/agents/racecar` :doc:`../../components_of_environments/agents/doggo` :doc:`../../components_of_environments/agents/ant`

在基于视觉的算法上，识别并记忆给定的模式是十分重要的能力，在这个任务当中，一个针对视觉记忆的开放性问题被提出。

针对问题的特殊性，Fading被具象化为FadingEasy和FadingHard两个类别，展现出更为合理的难度梯度，同时，精心设计的任务设定，允许在Easy和Hard之间，以及同一任务内部的不同level之间有意义的跨维度比较。

.. Note::

    在FadingEasy任务当中，给定的物体将在刷新后的 ``150steps`` 内线性地变得透明。

    刷新的条件：

        - Goal
            ``step = 0`` or ``goal_achieved = True``.
        - Obstacles
            ``step = 0`` or ``cost > 0``.

Rewards
-------

 - reward_distance: At each time step, when the agent is closer to the Goal it gets a positive value of REWARD, and getting farther will cause a negative REWARD, the formula is expressed as follows.

 .. math:: r_t = (D_{last} - D_{now})\beta

 Obviously when :math:`D_{last} > D_{now}`, :math:`r_t>0`. Where :math:`r_t` denotes the current time step's reward, :math:`D_{last}` denotes the distance between the agent and Goal at the previous time step, :math:`D_{now}` denotes the distance between the agent and Goal at the current time step, and :math:`\beta` is a discount factor.


 - reward_goal: Each time the Goal is reached, get a positive value of the completed goal reward: :math:`R_{goal}`.

Episode End
-----------

- When episode length is greater than 1000: ``Trucated = True``.

.. _FadingEasy0:


Level0
------

.. image:: ../../_static/images/fading_easy0.gif
    :align: center
    :scale: 100 %

Agent需要在信息消失的干扰下尽可能多地到达Goal的位置。


Fading Objects
^^^^^^^^^^^^^^

    - Goal

Costs
^^^^^

Nothing.

Randomness
^^^^^^^^^^

+--------------------------------+-------------------------+---------------+
| Scope                          | Range                   | Distribution  |
+================================+=========================+===============+
| rotation of agent and objects  | :math:`[0, 2\pi]`       | uniform       |
+--------------------------------+-------------------------+---------------+
| location of agent and objects  | :math:`[-1, -1, 1, 1]`  | uniform       |
+--------------------------------+-------------------------+---------------+

.. _FadingEasy1:

Level1
------

.. image:: ../../_static/images/fading_easy1.gif
    :align: center
    :scale: 100 %

Agent需要在信息消失的干扰下尽可能多地到达Goal的位置，同时避免进入Hazards的范围内，Vases=1但并不参与cost的计算。


Fading Objects
^^^^^^^^^^^^^^

    - Goal



Costs
^^^^^

.. list-table::
   :header-rows: 1

   * - Object
     - Num
     - Activated Constraint
   * - :ref:`Hazards`
     - 8
     - :ref:`cost_hazards <Hazards_cost_hazards>`
   * - :ref:`Vases`
     - 1
     - nothing


Randomness
^^^^^^^^^^

+--------------------------------+---------------------------------+---------------+
| Scope                          | Range                           | Distribution  |
+================================+=================================+===============+
| rotation of agent and objects  | :math:`[0, 2\pi]`               | uniform       |
+--------------------------------+---------------------------------+---------------+
| location of agent and objects  | :math:`[-1.5, -1.5, 1.5, 1.5]`  | uniform       |
+--------------------------------+---------------------------------+---------------+

.. _FadingEasy2:

Level2
------


.. image:: ../../_static/images/fading_easy2.gif
    :align: center
    :scale: 100 %

Agent需要在信息消失的干扰下尽可能多地到达Goal的位置，同时避免进入Hazards的范围内以及与Vases发生碰撞。

Fading Objects
^^^^^^^^^^^^^^

    - Goal
    - Hazards

Costs
^^^^^

.. list-table::
   :header-rows: 1

   * - Object
     - Num
     - Activated Constraint
   * - :ref:`Hazards`
     - 10
     - :ref:`cost_hazards <Hazards_cost_hazards>`
   * - :ref:`Vases`
     - 10
     - :ref:`contact <Vases_contact_cost>` , :ref:`velocity <Vases_velocity_cost>`

Randomness
^^^^^^^^^^

+--------------------------------+-------------------------+---------------+
| Scope                          | Range                   | Distribution  |
+================================+=========================+===============+
| rotation of agent and objects  | :math:`[0, 2\pi]`       | uniform       |
+--------------------------------+-------------------------+---------------+
| location of agent and objects  | :math:`[-2, -2, 2, 2]`  | uniform       |
+--------------------------------+-------------------------+---------------+
