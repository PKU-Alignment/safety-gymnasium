Car
===

.. list-table::

    * - .. figure:: ../../_static/images/car_front.jpeg
            :width: 200px
        .. centered:: front
      - .. figure:: ../../_static/images/car_back.jpeg
            :width: 200px
        .. centered:: back
      - .. figure:: ../../_static/images/car_left.jpeg
            :width: 200px
        .. centered:: left
      - .. figure:: ../../_static/images/car_right.jpeg
            :width: 200px
        .. centered:: right

A slightly more complex robot, moving in three dimensions, has two independently driven parallel wheels and one free-rolling rear wheel. For this robot, both steering and forward/backward movement require coordination of the two drives. It is similar in design to a simple robot used for educational purposes.

+---------------------------------+--------------------------------+
| **Specific Action Space**       | Box(-1.0, 1.0, (2,), float64)  |
+---------------------------------+--------------------------------+
| **Specific Observation Shape**  | (24,)                          |
+---------------------------------+--------------------------------+
| **Specific Observation High**   | inf                            |
+---------------------------------+--------------------------------+
| **Specific Observation Low**    | -inf                           |
+---------------------------------+--------------------------------+


Specific Action Space
---------------------

+------+---------------------------+--------------+--------------+-----------------------------------+-------------+------------+
| Num  | Action                    | Control Min  | Control Max  | Name (in corresponding XML file)  | Joint/Site  | Unit       |
+======+===========================+==============+==============+===================================+=============+============+
| 0    | To applied on left wheel  | 0            | 1            | left                              | hinge       | Force (N)  |
+------+---------------------------+--------------+--------------+-----------------------------------+-------------+------------+
| 1    | To applied on right wheel | 0            | 1            | right                             | hinge       | Force (N)  |
+------+---------------------------+--------------+--------------+-----------------------------------+-------------+------------+


Specific Observation Space
--------------------------

+-------+-----------------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| Size  | Observation                                                                 | Min  | Max  | Name (in corresponding XML file)  | Joint/Site  | Unit                       |
+=======+=============================================================================+======+======+===================================+=============+============================+
| 9     | Quaternions of the rear wheel which are turned into 3x3 rotation matrices.  | -inf | inf  | ballquat_rear                     | ball        | unitless                   |
+-------+-----------------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | Angle velocity of the rear wheel.                                           | -inf | inf  | ballangvel_rear                   | ball        | anglular velocity (rad/s)  |
+-------+-----------------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | accelerometer                                                               | -inf | inf  | accelerometer                     | site        | acceleration (m/s^2)       |
+-------+-----------------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | velocimeter                                                                 | -inf | inf  | velocimeter                       | site        | velocity (m/s)             |
+-------+-----------------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | gyro                                                                        | -inf | inf  | gyro                              | site        | anglular velocity (rad/s)  |
+-------+-----------------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | magnetometer                                                                | -inf | inf  | magnetometer                      | site        | magnetic flux (Wb)         |
+-------+-----------------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+


Specific Starting Randomness
----------------------------

Nothing.

Specific Episode End
--------------------

Nothing.
