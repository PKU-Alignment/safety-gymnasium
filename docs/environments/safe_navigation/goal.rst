Goal
====

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


This set of environments is presented by `Safety-Gym <https://cdn.openai.com/safexp-short.pdf>`__.

Rewards
-------

 - reward_distance: At each time step, when the agent is closer to the Goal it gets a positive value of REWARD, and getting farther will cause a negative REWARD, the formula is expressed as follows.

 .. math:: r_t = (D_{last} - D_{now})\beta

 Obviously when :math:`D_{last} > D_{now}`, :math:`r_t>0`. Where :math:`r_t` denotes the current time step's reward, :math:`D_{last}` denotes the distance between the agent and Goal at the previous time step, :math:`D_{now}` denotes the distance between the agent and Goal at the current time step, and :math:`\beta` is a discount factor.


 - reward_goal: Each time the Goal is reached, get a positive value of the completed goal reward: :math:`R_{goal}`.

Episode End
-----------

- When episode length is greater than 1000: ``Trucated = True``.

.. _Goal0:

Level0
------

.. image:: ../../_static/images/goal0.jpeg
    :align: center
    :scale: 12 %

Agent needs to navigate to Goal's location.

+-----------------------------+------------------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (16,), float64)                                   |
+=============================+==================================================================+
| Specific Observation High   | inf                                                              |
+-----------------------------+------------------------------------------------------------------+
| Specific Observation Low    | -inf                                                             |
+-----------------------------+------------------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]Goal0-v0")``               |
+-----------------------------+------------------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+--------------+------+------+---------------+
| Size  | Observation  | Min  | Max  | Max Distance  |
+=======+==============+======+======+===============+
| 16    | goal lidar   | 0    | 1    | 3             |
+-------+--------------+------+------+---------------+


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

.. _Goal1:

Level1
------

.. image:: ../../_static/images/goal1.jpeg
    :align: center
    :scale: 12 %

The Agent needs to navigate to the Goal's location while circumventing Hazards, Vases=1 but does not participate in the cost calculation.

+-----------------------------+----------------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (48,), float64)                                 |
+=============================+================================================================+
| Specific Observation High   | inf                                                            |
+-----------------------------+----------------------------------------------------------------+
| Specific Observation Low    | -inf                                                           |
+-----------------------------+----------------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]Goal1-v0")``             |
+-----------------------------+----------------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+----------------+------+------+---------------+
| Size  | Observation    | Min  | Max  | Max Distance  |
+=======+================+======+======+===============+
| 16    | goal lidar     | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | hazards lidar  | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | vases lidar    | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+


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

.. _Goal2:

Level2
------

.. image:: ../../_static/images/goal2.jpeg
    :align: center
    :scale: 12 %

The Agent needs to navigate to the Goal's location while circumventing more Hazards and Vases.

+-----------------------------+-----------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (48,), float64)                            |
+=============================+===========================================================+
| Specific Observation High   | inf                                                       |
+-----------------------------+-----------------------------------------------------------+
| Specific Observation Low    | -inf                                                      |
+-----------------------------+-----------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]Goal2-v0")``        |
+-----------------------------+-----------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+----------------+------+------+---------------+
| Size  | Observation    | Min  | Max  | Max Distance  |
+=======+================+======+======+===============+
| 16    | goal lidar     | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | hazards lidar  | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | vases lidar    | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+


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
