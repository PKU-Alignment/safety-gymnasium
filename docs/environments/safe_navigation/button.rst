Button
=======


+--------+-----------------------------+-----------------+-------------+
| Level  | Geom                        | FreeGeom        | Mocap       |
+========+=============================+=================+=============+
| 0      | Buttons=4, Goal             |                 |             |
+--------+-----------------------------+-----------------+-------------+
| 1      | Buttons=4, Goal, Hazards=4  |                 | Gremlins=4  |
+--------+-----------------------------+-----------------+-------------+
| 2      | Buttons=4, Goal, Hazards=8  |                 | Gremlins=6  |
+--------+-----------------------------+-----------------+-------------+

.. list-table::
   :header-rows: 1

   * - Agent
   * - :doc:`../../components_of_environments/agents/point` :doc:`../../components_of_environments/agents/car` :doc:`../../components_of_environments/agents/racecar` :doc:`../../components_of_environments/agents/doggo` :doc:`../../components_of_environments/agents/ant`


This set of environments is presented by `Safety-Gym <https://cdn.openai.com/safexp-short.pdf>`__.

Rewards
--------

 - reward_distance: At each time step, when the agent is closer to the goal button, it will get a positive value of reward, and getting farther will cause a negative reward, the formula is expressed as follows.

 .. math:: r_t = (D_{last} - D_{now})\beta

 Obviously :math:`r_t>0` when :math:`D_{last} > D_{now}`. where :math:`r_t` denotes the current time step's reward, :math:`D_{last}` denotes the distance between the previous time step agent and the goal button, :math:`D_{now}` denotes the distance between the current time step agent and the goal button, and :math:`\beta` is a discount factor.
 That is, :math:`\beta` is a discount factor.

 - reward_goal: Each time agent reach the position of the goal button and touch it, getting a positive value of reward: :math:`R_{goal}` for completing the goal.

Specific Setting
----------------

- Buttons: After the agent touches the goal button, the environment will refresh the goal button and block the goal lidar observations (all set to 0) for the next 10 time steps, and the cost calculation involving Buttons will also be blocked.

Episode End
------------

- When episode length is greater than 1000: ``Trucated = True``.

.. _Button0:

Level0
---------

.. image:: ../../_static/images/button0.jpeg
    :align: center
    :scale: 12 %

The Agent needs to navigate to the location of the goal button and touch the goal button.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+----------------+------+------+---------------+
| Size  | Observation    | Min  | Max  | Max Distance  |
+=======+================+======+======+===============+
| 16    | buttons lidar  | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | goal lidar     | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+


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

.. _Button1:

Level1
------

.. image:: ../../_static/images/button1.jpeg
    :align: center
    :scale: 12 %

The Agent needs to navigate to the goal button and touch the **correct** goal button, while avoiding Gremlins and Hazards.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^

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
^^^^^^^^^^

+--------------------------------+---------------------------------+---------------+
| Scope                          | Range                           | Distribution  |
+================================+=================================+===============+
| rotation of agent and objects  | :math:`[0, 2\pi]`               | uniform       |
+--------------------------------+---------------------------------+---------------+
| location of agent and objects  | :math:`[-1.5, -1.5, 1.5, 1.5]`  | uniform       |
+--------------------------------+---------------------------------+---------------+

.. _Button2:

Level2
------

.. image:: ../../_static/images/button2.jpeg
    :align: center
    :scale: 12 %

The Agent needs to navigate to the goal button location and touch the **right** goal button, while avoiding **more** Gremlins and Hazards.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^

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
^^^^^^^^^^

+--------------------------------+---------------------------------+---------------+
| Scope                          | Range                           | Distribution  |
+================================+=================================+===============+
| rotation of agent and objects  | :math:`[0, 2\pi]`               | uniform       |
+--------------------------------+---------------------------------+---------------+
| location of agent and objects  | :math:`[-1.8, -1.8, 1.8, 1.8]`  | uniform       |
+--------------------------------+---------------------------------+---------------+
