Point
=====

.. _Point:

.. list-table::

    * - .. figure:: ../../_static/images/point_front.jpeg
            :width: 200px
        .. centered:: front
      - .. figure:: ../../_static/images/point_back.jpeg
            :width: 200px
        .. centered:: back
      - .. figure:: ../../_static/images/point_left.jpeg
            :width: 200px
        .. centered:: left
      - .. figure:: ../../_static/images/point_right.jpeg
            :width: 200px
        .. centered:: right

A simple robot constrained to a 2D plane has two actuators, one for rotation and the other for forward/backward movement. This decomposed control scheme makes it particularly easy to control the robot's navigation. It has a small square in front of it, which makes it easier to visually determine the robot's orientation and also helps Point push the box that appears in Push.

+---------------------------------+--------------------------------+
| **Specific Action Space**       | Box(-1.0, 1.0, (2,), float64)  |
+---------------------------------+--------------------------------+
| **Specific Observation Shape**  | (12,)                          |
+---------------------------------+--------------------------------+
| **Specific Observation High**   | inf                            |
+---------------------------------+--------------------------------+
| **Specific Observation Low**    | -inf                           |
+---------------------------------+--------------------------------+


Specific Action Space
---------------------

+------+---------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+-----------------+
| Num  | Action                                                  | Control Min  | Control Max  | Name (in corresponding XML file)  | Joint/Site  | Unit            |
+======+=========================================================+==============+==============+===================================+=============+=================+
| 0    | force applied on the agent to move forward or backward  | 0            | 1            | x                                 | site        | force (N)       |
+------+---------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+-----------------+
| 1    | velocity of the agent, which is around the z-axis.      | 0            | 1            | z                                 | hinge       | velocity (m/s)  |
+------+---------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+-----------------+


Specific Observation Space
--------------------------

+-------+----------------+------+------+-----------------------------------+-------------+----------------------------+
| Size  | Observation    | Min  | Max  | Name (in corresponding XML file)  | Joint/Site  | Unit                       |
+=======+================+======+======+===================================+=============+============================+
| 3     | accelerometer  | -inf | inf  | accelerometer                     | site        | acceleration (m/s^2)       |
+-------+----------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | velocimeter    | -inf | inf  | velocimeter                       | site        | velocity (m/s)             |
+-------+----------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | gyro           | -inf | inf  | gyro                              | site        | anglular velocity (rad/s)  |
+-------+----------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | magnetometer   | -inf | inf  | magnetometer                      | site        | magnetic flux (Wb)         |
+-------+----------------+------+------+-----------------------------------+-------------+----------------------------+


Specific Starting Randomness
----------------------------

Nothing.

Specific Episode End
--------------------

Nothing.
