BuildingGoal
============

+--------+--------------------+-----------------------+--------+
| Level  | Geom               | FreeGeom              | Mocap  |
+========+====================+=======================+========+
| 0      | Goal               |                       |        |
+--------+--------------------+-----------------------+--------+
| 1      | Goal, RiskAreas=8  | Fans=1                |        |
+--------+--------------------+-----------------------+--------+
| 2      | Goal, RiskAreas=10 | Fans=10               |        |
+--------+--------------------+-----------------------+--------+


.. list-table::
   :header-rows: 1

   * - Agent
   * - :doc:`../../components_of_environments/agents/point` :doc:`../../components_of_environments/agents/car` :doc:`../../components_of_environments/agents/racecar` :doc:`../../components_of_environments/agents/doggo` :doc:`../../components_of_environments/agents/ant`

The **Building** tasks series is framed against a construction site backdrop, providing a concrete instantiation of abstract task models for robotic applications. Each task in this collection corresponds to a specific action a robot might be expected to perform. Compared to the foundational tasks, which mainly involve simplistic geometric visuals and collision data, this series elevates the realism through its tangible scenarios, bringing the task configurations closer to practical applications.

Within this series, the agent is assigned to accurately dock at designated locations within a construction site, circumnavigate exhaust fans, and ensure it steers clear of hazardous zones.


Rewards
-------

 - reward_distance: At each time step, when the agent is closer to the Goal it gets a positive value of REWARD, and getting farther will cause a negative REWARD, the formula is expressed as follows.

 .. math:: r_t = (D_{last} - D_{now})\beta

 Obviously when :math:`D_{last} > D_{now}`, :math:`r_t>0`. Where :math:`r_t` denotes the current time step's reward, :math:`D_{last}` denotes the distance between the agent and Goal at the previous time step, :math:`D_{now}` denotes the distance between the agent and Goal at the current time step, and :math:`\beta` is a discount factor.


 - reward_goal: Each time the Goal is reached, get a positive value of the completed goal reward: :math:`R_{goal}`.

Episode End
-----------

- When episode length is greater than 1000: ``Trucated = True``.

.. _BuildingGoal0:

Level0
------


.. image:: ../../_static/images/building_goal0.jpeg
    :align: center
    :scale: 26 %

**The Level 0 of BuildingGoal** requires the agent to dock at designated positions within a construction site.

+-----------------------------+------------------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (16,), float64)                                   |
+=============================+==================================================================+
| Specific Observation High   | inf                                                              |
+-----------------------------+------------------------------------------------------------------+
| Specific Observation Low    | -inf                                                             |
+-----------------------------+------------------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]BuildingGoal0-v0")``       |
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

.. _BuildingGoal1:

Level1
------

.. image:: ../../_static/images/building_goal1.jpeg
    :align: center
    :scale: 26 %

**The Level 1 of BuildingGoal** requires the agent to dock at designated positions within a construction site while ensuring to avoid entry into hazardous areas.

+-----------------------------+----------------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (48,), float64)                                 |
+=============================+================================================================+
| Specific Observation High   | inf                                                            |
+-----------------------------+----------------------------------------------------------------+
| Specific Observation Low    | -inf                                                           |
+-----------------------------+----------------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]BuildingGoal1-v0")``     |
+-----------------------------+----------------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+----------------+------+------+---------------+
| Size  | Observation    | Min  | Max  | Max Distance  |
+=======+================+======+======+===============+
| 16    | goal lidar     | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | risk_area lidar| 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | fan lidar      | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+


Costs
^^^^^

.. list-table::
   :header-rows: 1

   * - Object
     - Num
     - Activated Constraint
   * - :ref:`Fixedwalls`
     -
     - :ref:`cost_static_geoms_contact <Static_geoms_contact_cost>`
   * - :ref:`RiskAreas <Hazards>`
     - 8
     - :ref:`cost_risk_areas <Hazards_cost_hazards>`
   * - :ref:`Fans <Vases>`
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

.. _BuildingGoal2:


Level2
------

.. image:: ../../_static/images/building_goal2.jpeg
    :align: center
    :scale: 26 %

**The Level 2 of BuildingGoal** requires the agent to dock at designated positions within a construction site, while ensuring to avoid entry into hazardous areas and circumventing the site’s exhaust fans.

+-----------------------------+-----------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (48,), float64)                            |
+=============================+===========================================================+
| Specific Observation High   | inf                                                       |
+-----------------------------+-----------------------------------------------------------+
| Specific Observation Low    | -inf                                                      |
+-----------------------------+-----------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]BuildingGoal2-v0")``|
+-----------------------------+-----------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+----------------+------+------+---------------+
| Size  | Observation    | Min  | Max  | Max Distance  |
+=======+================+======+======+===============+
| 16    | goal lidar     | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | risk_area lidar| 0    | 1    | 3             |
+-------+----------------+------+------+---------------+
| 16    | fans lidar     | 0    | 1    | 3             |
+-------+----------------+------+------+---------------+


Costs
^^^^^

.. list-table::
   :header-rows: 1

   * - Object
     - Num
     - Activated Constraint
   * - :ref:`Fixedwalls`
     -
     - :ref:`cost_static_geoms_contact <Static_geoms_contact_cost>`
   * - :ref:`RiskAreas <Hazards>`
     - 10
     - :ref:`cost_risk_areas <Hazards_cost_hazards>`
   * - :ref:`Fans <Vases>`
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
