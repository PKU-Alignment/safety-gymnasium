Doggo
=====

.. list-table::

    * - .. figure:: ../../_static/images/doggo_front.jpeg
            :width: 200px
        .. centered:: front
      - .. figure:: ../../_static/images/doggo_back.jpeg
            :width: 200px
        .. centered:: back
      - .. figure:: ../../_static/images/doggo_left.jpeg
            :width: 200px
        .. centered:: left
      - .. figure:: ../../_static/images/doggo_right.jpeg
            :width: 200px
        .. centered:: right

Doggo is a quadrupedal robot with bilateral symmetry. Each of the four legs has two controls at the hip, for azimuth and elevation relative to the torso, and one in the knee, controlling angle. It is designed such that a uniform random policy should keep the robot from falling over and generating some travel.

+---------------------------------+--------------------------------+
| **Specific Action Space**       | Box(-1.0, 1.0, (12,), float64) |
+---------------------------------+--------------------------------+
| **Specific Observation Shape**  | (12,)                          |
+---------------------------------+--------------------------------+
| **Specific Observation High**   | inf                            |
+---------------------------------+--------------------------------+
| **Specific Observation Low**    | -inf                           |
+---------------------------------+--------------------------------+


Specific Action Space
---------------------

+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| Num | Action                                                                          | Control Min | Control Max | Name (in corresponding XML file) | Joint/Site | Unit         |
+=====+=================================================================================+=============+=============+==================================+============+==============+
| 0   | torque applied on the rotor between the torso and front left hip around z-axis  | -1          | 1           | hip_1_z                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| 1   | torque applied on the rotor between the torso and back left hip around z-axis   | -1          | 1           | hip_2_z                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| 2   | torque applied on the rotor between the torso and back right hip around z-axis  | -1          | 1           | hip_3_z                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| 3   | torque applied on the rotor between the torso and front right hip around z-axis | -1          | 1           | hip_4_z                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| 4   | torque applied on the rotor between the torso and front left hip around y-axis  | -1          | 1           | hip_1_y                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| 5   | torque applied on the rotor between the torso and back left hip around y-axis   | -1          | 1           | hip_2_y                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| 6   | torque applied on the rotor between the torso and back right hip around y-axis  | -1          | 1           | hip_3_y                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| 7   | torque applied on the rotor between the torso and front right hip around y-axis | -1          | 1           | hip_4_y                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| 8   | torque applied on the rotor between the front left two links                    | -1          | 1           | ankle_1                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| 9   | torque applied on the rotor between the back left two links                     | -1          | 1           | ankle_2                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| 10  | torque applied on the rotor between the back right two links                    | -1          | 1           | ankle_3                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+
| 11  | torque applied on the rotor between the front right two links                   | -1          | 1           | ankle_4                          | hinge      | torque (N m) |
+-----+---------------------------------------------------------------------------------+-------------+-------------+----------------------------------+------------+--------------+


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
