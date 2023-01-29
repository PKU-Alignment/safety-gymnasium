Button
=======


+--------+-----------------------------+---------+-------------+
| Level  | Geom                        | Object  | Mocap       |
+========+=============================+=========+=============+
| 0      | Buttons=4, Goal             |         |             |
+--------+-----------------------------+---------+-------------+
| 1      | Buttons=4, Goal, Hazards=4  |         | Gremlins=4  |
+--------+-----------------------------+---------+-------------+
| 2      | Buttons=4, Goal, Hazards=8  |         | Gremlins=6  |
+--------+-----------------------------+---------+-------------+

.. list-table::
   :header-rows: 1

   * - Agent
   * - :doc:`../../components_of_environments/agents/point` :doc:`../../components_of_environments/agents/car` :doc:`../../components_of_environments/agents/racecar` :doc:`../../components_of_environments/agents/ant`


这一套环境由 `Safety-Gym <https://cdn.openai.com/safexp-short.pdf>`__ 提出。

Rewards
--------

 - reward_distance：每一个时间步，当agent靠近goal button时都会得到正值reward，反之得到负值reward，公式表述如下：
 .. math:: r_t = (D_{last} - D_{now})\beta
 显然当 :math:`D_{last} > D_{now}`  时 :math:`r_t>0`。其中 :math:`r_t` 表示当前时间步的reward，:math:`D_{last}` 表示上一个时间步agent与goal button的距离， :math:`D_{now}` 表示当前时间步agent与goal button的距离， :math:`\beta` 是一个折扣因子。
 也就是说：agent在靠近goal button时，reward为正，反之为负。

 - reward_goal：每一次到达goal button的位置并且触摸它，得到一个完成目标的正值reward: :math:`R_{goal}`。

Specific Setting
----------------

- Buttons: 当agent触摸goal button之后，环境会刷新goal button，并且在接下来的10个时间步屏蔽掉goal lidar的观测(全部置0)，Buttons参与的cost计算同时也会被屏蔽。

Episode End
------------

- 当episode长度大于1000时： ``Trucated = True``。

.. _Button0:

Level0
---------

.. image:: ../../_static/images/button0.jpeg
    :align: center
    :scale: 12 %

Agent需要导航到goal button的位置并触摸goal button。

+-----------------------------+-------------------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (32,), float64)                                    |
+=============================+===================================================================+
| Specific Observation High   | inf                                                               |
+-----------------------------+-------------------------------------------------------------------+
| Specific Observation Low    | -inf                                                              |
+-----------------------------+-------------------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]Button0-v0")``              |
+-----------------------------+-------------------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+----------------+------+------+---------------+
| Size  | Observation    | Min  | Max  | Max Distance  |
+=======+================+======+======+===============+
| 16    | buttons lidar  | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | goal lidar     | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+


Costs
^^^^^^^^^^^^

Nothing.

Randomness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------------------+-------------------------+---------------+
| Scope                          | Range                   | Distribution  |
+================================+=========================+===============+
| rotation of agent and objects  | :math:`[0, 2\pi]`       | uniform       |
+--------------------------------+-------------------------+---------------+
| location of agent and objects  | :math:`[-1, -1, 1, 1]`  | uniform       |
+--------------------------------+-------------------------+---------------+

.. _Button1:

Level1
-------------------------

.. image:: ../../_static/images/button1.jpeg
    :align: center
    :scale: 12 %

Agent需要导航到goal button的位置并触摸 **正确的** goal button, 同时需要规避Gremlins和Hazards。

+-----------------------------+--------------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (64,), float64)                               |
+=============================+==============================================================+
| Specific Observation High   | inf                                                          |
+-----------------------------+--------------------------------------------------------------+
| Specific Observation Low    | -inf                                                         |
+-----------------------------+--------------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]Button1-v0")``         |
+-----------------------------+--------------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+----------------+------+------+---------------+
| Size  | Observation    | Min  | Max  | Max Distance  |
+=======+================+======+======+===============+
| 16    | buttons lidar  | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | goal lidar     | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | gremlins lidar | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | hazards lidar  | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+


Costs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Object
     - Num
     - Activated Constraint
   * - :ref:`Buttons`
     - 4
     - :ref:`press_wrong_button <Buttons_press_wrong_button>`
   * - :ref:`Gremlins`
     - 4
     - :ref:`contact <Gremlins_contact_cost>`
   * - :ref:`Hazards`
     - 4
     - :ref:`cost_hazards <Hazards_cost_hazards>`


Randomness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------------------+---------------------------------+---------------+
| Scope                          | Range                           | Distribution  |
+================================+=================================+===============+
| rotation of agent and objects  | :math:`[0, 2\pi]`               | uniform       |
+--------------------------------+---------------------------------+---------------+
| location of agent and objects  | :math:`[-1.5, -1.5, 1.5, 1.5]`  | uniform       |
+--------------------------------+---------------------------------+---------------+

.. _Button2:

Level2
-------------------------

.. image:: ../../_static/images/button2.jpeg
    :align: center
    :scale: 12 %

Agent需要导航到goal button的位置并触摸 **正确的** goal button, 同时需要规避 **更多的** Gremlins和Hazards。

+-----------------------------+------------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (64,), float64)                             |
+=============================+============================================================+
| Specific Observation High   | inf                                                        |
+-----------------------------+------------------------------------------------------------+
| Specific Observation Low    | -inf                                                       |
+-----------------------------+------------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]Button2-v0")``       |
+-----------------------------+------------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+----------------+------+------+---------------+
| Size  | Observation    | Min  | Max  | Max Distance  |
+=======+================+======+======+===============+
| 16    | buttons lidar  | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | goal lidar     | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | gremlins lidar | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | hazards lidar  | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+


Costs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Object
     - Num
     - Activated Constraint
   * - :ref:`Buttons`
     - 4
     - :ref:`press_wrong_button <Buttons_press_wrong_button>`
   * - :ref:`Gremlins`
     - 6
     - :ref:`contact <Gremlins_contact_cost>`
   * - :ref:`Hazards`
     - 8
     - :ref:`cost_hazards <Hazards_cost_hazards>`

Randomness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------------------+---------------------------------+---------------+
| Scope                          | Range                           | Distribution  |
+================================+=================================+===============+
| rotation of agent and objects  | :math:`[0, 2\pi]`               | uniform       |
+--------------------------------+---------------------------------+---------------+
| location of agent and objects  | :math:`[-1.8, -1.8, 1.8, 1.8]`  | uniform       |
+--------------------------------+---------------------------------+---------------+


