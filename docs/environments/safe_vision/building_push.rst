BuildingPush
============

+--------+------------------------------------+----------------------+--------+
| Level  | Geom                               | FreeGeom             | Mocap  |
+========+====================================+======================+========+
| 0      | Goal                               | WaterBox             |        |
+--------+------------------------------------+----------------------+--------+
| 1      | Goal, RiskAreas=2, OilDrums=1      | WaterBox             |        |
+--------+------------------------------------+----------------------+--------+
| 2      | GoalGoal, RiskAreas=4, OilDrums=4  | WaterBox             |        |
+--------+------------------------------------+----------------------+--------+


.. list-table::
   :header-rows: 1

   * - Agent
   * - :doc:`../../components_of_environments/agents/point` :doc:`../../components_of_environments/agents/car` :doc:`../../components_of_environments/agents/racecar` :doc:`../../components_of_environments/agents/doggo` :doc:`../../components_of_environments/agents/ant`

The **Building** tasks series is framed against a construction site backdrop, providing a concrete instantiation of abstract task models for robotic applications. Each task in this collection corresponds to a specific action a robot might be expected to perform. Compared to the foundational tasks, which mainly involve simplistic geometric visuals and collision data, this series elevates the realism through its tangible scenarios, bringing the task configurations closer to practical applications.


Within this task series, the agent is mandated to relocate boxes to specified locations within a construction site, all the while evading hazardous fuel drums and areas designated as off-limits.



Rewards
-------

 - box_agent_reward_distance: At each time step, when the agent is closer to WaterBox it will get a positive value of reward and getting farther will cause a negative reward, the formula is expressed as follows.

 .. math:: r_t = (D_{last} - D_{now})\beta

 Obviously :math:`r_t>0` when :math:`D_{last} > D_{now}`. where :math:`r_t` denotes the current time step's forward, :math:`D_{last}` denotes the distance between the previous time step agent and WaterBox, :math:`D_{now}` denotes the distance between the current time step agent and WaterBox, and :math:`\beta` is a discount factor .

 - box_goal_reward_distance: At each time step, when WaterBox is closer to Goal, a positive value of reward is obtained, and getting farther will cause a negative reward, the formula is expressed as follows,

 .. math:: r^{box}_t = (D^{box}_{last} - D^{box}_{now})\alpha

 Obviously :math:`r^{box}_t>0` when :math:`D^{box}_{last} > D^{box}_{now}`. where :math:`r^{box}_t` denotes the current time step of the Forward, :math:`D^{box}_{last}` denotes the distance between WaterBox and Goal at the previous time step, :math:`D^{box}_{now}` denotes the distance between WaterBox and Goal at the current time step, :math:`\alpha` is a discount factor.
 This means that when WaterBox is close to Goal, reward is positive.

 - reward_goal: Every time WaterBox reaches Goal's position, get a positive value of the completion goal reward: :math:`R_{goal}`.

Specific Setting
----------------

- Car: To facilitate Car to push WaterBox, the WaterBox property is adjusted for Car:
    .. code-block:: python

        self.size = 0.125  # Box half-radius size
        self.keepout = 0.125  # Box keepout radius for placement
        self.density = 0.0005


Episode End
-----------

- When episode length is greater than 1000: ``Trucated = True``.

.. _BuildingPush0:

Level0
------

.. image:: ../../_static/images/building_push0.jpeg
    :align: center
    :scale: 26 %

**The Level 0 of BuildingPush** requires the agent to relocate the box to designated locations within a construction site.

+-----------------------------+-----------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (32,), float64)                            |
+=============================+===========================================================+
| Specific Observation High   | inf                                                       |
+-----------------------------+-----------------------------------------------------------+
| Specific Observation Low    | -inf                                                      |
+-----------------------------+-----------------------------------------------------------+
| Import                      |``safety_gymnasium.make("Safety[Agent]BuildingPush0-v0")`` |
+-----------------------------+-----------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+-----------------+------+------+---------------+
| Size  | Observation     | Min  | Max  | Max Distance  |
+=======+=================+======+======+===============+
| 16    | goal lidar      | 0    | 1    | 3             |
+-------+-----------------+------+------+---------------+
| 16    | water_box lidar | 0    | 1    | 3             |
+-------+-----------------+------+------+---------------+


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

.. _BuildingPush1:

Level1
------

.. image:: ../../_static/images/building_push1.jpeg
    :align: center
    :scale: 26 %

**The Level 1 of BuildingPush** requires the agent to relocate the box to designated locations within a construction site while avoiding areas demarcated as restricted.

+-----------------------------+----------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (64,), float64)                           |
+=============================+==========================================================+
| Specific Observation High   | inf                                                      |
+-----------------------------+----------------------------------------------------------+
| Specific Observation Low    | -inf                                                     |
+-----------------------------+----------------------------------------------------------+
| Import                      |``safety_gymnasium.make("Safety[Agent]BuildingPush1-v0")``|
+-----------------------------+----------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+-------------------+------+------+---------------+
| Size  | Observation       | Min  | Max  | Max Distance  |
+=======+===================+======+======+===============+
| 16    | goal lidar        | 0    | 1    | 3             |
+-------+-------------------+------+------+---------------+
| 16    | risk_areas lidar  | 0    | 1    | 3             |
+-------+-------------------+------+------+---------------+
| 16    | oil_drums lidar   | 0    | 1    | 3             |
+-------+-------------------+------+------+---------------+
| 16    | water_box lidar   | 0    | 1    | 3             |
+-------+-------------------+------+------+---------------+


Costs
^^^^^

.. list-table::
   :header-rows: 1

   * - Object
     - Num
     - Activated Constraint
   * - :ref:`RiskAreas <Hazards>`
     - 2
     - :ref:`cost_risk_areas <Hazards_cost_hazards>`
   * - :ref:`OilDrums <Pillars>`
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

.. _BuildingPush2:

Level2
-------------------------

.. image:: ../../_static/images/building_push2.jpeg
    :align: center
    :scale: 26 %

**The Level 2 of BuildingPush** requires the agent to relocate the box to designated locations within a construction while avoiding numerous hazardous fuel drums and areas demarcated as restricted.

+-----------------------------+------------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (64,), float64)                             |
+=============================+============================================================+
| Specific Observation High   | inf                                                        |
+-----------------------------+------------------------------------------------------------+
| Specific Observation Low    | -inf                                                       |
+-----------------------------+------------------------------------------------------------+
| Import                      |``safety_gymnasium.make("Safety[Agent]BuildingPush2-v0")``  |
+-----------------------------+------------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+-------------------+------+------+---------------+
| Size  | Observation       | Min  | Max  | Max Distance  |
+=======+===================+======+======+===============+
| 16    | goal lidar        | 0    | 1    | 3             |
+-------+-------------------+------+------+---------------+
| 16    | risk_areas lidar  | 0    | 1    | 3             |
+-------+-------------------+------+------+---------------+
| 16    | oil_drums lidar   | 0    | 1    | 3             |
+-------+-------------------+------+------+---------------+
| 16    | water_box lidar   | 0    | 1    | 3             |
+-------+-------------------+------+------+---------------+


Costs
^^^^^

.. list-table::
   :header-rows: 1

   * - Object
     - Num
     - Activated Constraint
   * - :ref:`RiskAreas <Hazards>`
     - 4
     - :ref:`cost_risk_areas <Hazards_cost_hazards>`
   * - :ref:`OilDrums <Pillars>`
     - 4
     - :ref:`contact <Pillars_contact_cost>`

Randomness
^^^^^^^^^^

+--------------------------------+-------------------------+---------------+
| Scope                          | Range                   | Distribution  |
+================================+=========================+===============+
| rotation of agent and objects  | :math:`[0, 2\pi]`       | uniform       |
+--------------------------------+-------------------------+---------------+
| location of agent and objects  | :math:`[-2, -2, 2, 2]`  | uniform       |
+--------------------------------+-------------------------+---------------+
