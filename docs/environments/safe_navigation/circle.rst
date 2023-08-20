Circle
==========

+--------+---------------------+-----------------+--------+
| Level  | Geom                | FreeGeom        | Mocap  |
+========+=====================+=================+========+
| 0      | Circle              |                 |        |
+--------+---------------------+-----------------+--------+
| 1      | Circle, Sigwalls=2  |                 |        |
+--------+---------------------+-----------------+--------+
| 2      | Circle, Sigwalls=4  |                 |        |
+--------+---------------------+-----------------+--------+


.. list-table::
   :header-rows: 1

   * - Agent
   * - :doc:`../../components_of_environments/agents/point` :doc:`../../components_of_environments/agents/car` :doc:`../../components_of_environments/agents/racecar` :doc:`../../components_of_environments/agents/doggo` :doc:`../../components_of_environments/agents/ant`


This set of environments is based on the idea of Humanoid-Circle in `Constrained Policy Optimization <https://arxiv.org/abs/1705.10528>`__.

Rewards
-------------------------



 - loop reward：

 .. math:: R_t = \frac{1}{1 + |r_{agent} - r_{circle}|} * \frac{(-uy + vx)}{r_{agent}}

 where :math:`R_t` is the current time-step of the redirection, :math:`u`, :math:`v` is the x-y axis velocity component of the agent, :math:`x` , :math:`y` are the x-y axis coordinates of the agent, :math:`r_{agent}` is the Euclidean distance of the agent from the origin, :math:`r_{circle}` is the radius of the Circle geometry. Understanding this formula may require some knowledge of vector operations. **Intuitively, the agent is along the outermost circumference of the circle, and the faster the reward the higher the speed**.


Episode End
-----------

- When episode length is greater than 500: ``Trucated = True``.

.. _Circle0:

Level0
------

.. image:: ../../_static/images/circle0.jpeg
    :align: center
    :scale: 12 %

Agent needs to circle around the center of the circle area.

+-----------------------------+----------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (16,), float64)                           |
+=============================+==========================================================+
| Specific Observation High   | inf                                                      |
+-----------------------------+----------------------------------------------------------+
| Specific Observation Low    | -inf                                                     |
+-----------------------------+----------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]Circle0-v0")``     |
+-----------------------------+----------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+--------------+------+------+---------------+
| Size  | Observation  | Min  | Max  | Max Distance  |
+=======+==============+======+======+===============+
| 16    | circle lidar | 0    | 1    | 6             |
+-------+--------------+------+------+---------------+


Costs
^^^^^

Nothing.

Randomness
^^^^^^^^^^

+--------------------+---------------------------------+---------------+
| Scope              | Range                           | Distribution  |
+====================+=================================+===============+
| rotation of agent  | :math:`[0, 2\pi]`               | uniform       |
+--------------------+---------------------------------+---------------+
| location of agent  | :math:`[-0.8, -0.8, 0.8, 0.8]`  | uniform       |
+--------------------+---------------------------------+---------------+

.. _Circle1:

Level1
-------------------------

.. image:: ../../_static/images/circle1.jpeg
    :align: center
    :scale: 12 %

Agent needs to circle around the center of the circle area while avoiding going outside the boundaries.

+-----------------------------+----------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (16,), float64)                           |
+=============================+==========================================================+
| Specific Observation High   | inf                                                      |
+-----------------------------+----------------------------------------------------------+
| Specific Observation Low    | -inf                                                     |
+-----------------------------+----------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]Circle1-v0")``     |
+-----------------------------+----------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+--------------+------+------+---------------+
| Size  | Observation  | Min  | Max  | Max Distance  |
+=======+==============+======+======+===============+
| 16    | circle lidar | 0    | 1    | 6             |
+-------+--------------+------+------+---------------+


Costs
^^^^^

.. list-table::
   :header-rows: 1

   * - Object
     - Num
     - Activated Constraint
   * - :ref:`Sigwalls`
     - 2
     - :ref:`out_of_boundary <Sigwalls_out_of_boundary_cost>`

Randomness
^^^^^^^^^^

+--------------------+---------------------------------+---------------+
| Scope              | Range                           | Distribution  |
+====================+=================================+===============+
| rotation of agent  | :math:`[0, 2\pi]`               | uniform       |
+--------------------+---------------------------------+---------------+
| location of agent  | :math:`[-0.8, -0.8, 0.8, 0.8]`  | uniform       |
+--------------------+---------------------------------+---------------+

.. _Circle2:

Level2
------

.. image:: ../../_static/images/circle2.jpeg
    :align: center
    :scale: 12 %

Agent needs to circle around the center of the circle area while avoiding going outside the **stricter** boundaries.

+-----------------------------+---------------------------------------------------------------+
| Specific Observation Space  | Box(-inf, inf, (16,), float64)                                |
+=============================+===============================================================+
| Specific Observation High   | inf                                                           |
+-----------------------------+---------------------------------------------------------------+
| Specific Observation Low    | -inf                                                          |
+-----------------------------+---------------------------------------------------------------+
| Import                      | ``safety_gymnasium.make("Safety[Agent]Circle2-v0")``          |
+-----------------------------+---------------------------------------------------------------+


Specific Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------+--------------+------+------+---------------+
| Size  | Observation  | Min  | Max  | Max Distance  |
+=======+==============+======+======+===============+
| 16    | circle lidar | 0    | 1    | 6             |
+-------+--------------+------+------+---------------+


Costs
^^^^^

.. list-table::
   :header-rows: 1

   * - Object
     - Num
     - Activated Constraint
   * - :ref:`Sigwalls`
     - 4
     - :ref:`out_of_boundary <Sigwalls_out_of_boundary_cost>`

Randomness
^^^^^^^^^^

+--------------------+---------------------------------+---------------+
| Scope              | Range                           | Distribution  |
+====================+=================================+===============+
| rotation of agent  | :math:`[0, 2\pi]`               | uniform       |
+--------------------+---------------------------------+---------------+
| location of agent  | :math:`[-0.8, -0.8, 0.8, 0.8]`  | uniform       |
+--------------------+---------------------------------+---------------+
