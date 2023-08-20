Ant
===

.. _Ant:

.. list-table::

    * - .. figure:: ../../_static/images/ant_front.jpeg
            :width: 200px
        .. centered:: front
      - .. figure:: ../../_static/images/ant_back.jpeg
            :width: 200px
        .. centered:: back
      - .. figure:: ../../_static/images/ant_left.jpeg
            :width: 200px
        .. centered:: left
      - .. figure:: ../../_static/images/ant_right.jpeg
            :width: 200px
        .. centered:: right


A quadrupedal robot, based on the model proposed in `High-Dimensional Continuous Control Using Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`__. Moving in three dimensions, it consists of a torso and four legs connected together, each leg consisting of two hinged connecting limbs, which are at the same time connected to the torso by hinges. It is necessary to coordinate the movement of the four legs in the target direction by applying moments to the drivers of the eight hinges.

+---------------------------------+--------------------------------+
| **Specific Action Space**       | Box(-1.0, 1.0, (8,), float64)  |
+---------------------------------+--------------------------------+
| **Specific Observation Shape**  | (40,)                          |
+---------------------------------+--------------------------------+
| **Specific Observation High**   | inf                            |
+---------------------------------+--------------------------------+
| **Specific Observation Low**    | -inf                           |
+---------------------------------+--------------------------------+


Specific Action Space
---------------------

+------+-------------------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+---------------+
| Num  | Action                                                            | Control Min  | Control Max  | Name (in corresponding XML file)  | Joint/Site  | Unit          |
+======+===================================================================+==============+==============+===================================+=============+===============+
| 0    | torque applied on the rotor between the torso and front left hip  | -1           | 1            | hip_1 (front_left_leg)            | hinge       | torque (N m)  |
+------+-------------------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+---------------+
| 1    | torque applied on the rotor between the front left two links      | -1           | 1            | angle_1 (front_left_leg)          | hinge       | torque (N m)  |
+------+-------------------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+---------------+
| 2    | torque applied on the rotor between the torso and front right hip | -1           | 1            | hip_2 (front_right_leg)           | hinge       | torque (N m)  |
+------+-------------------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+---------------+
| 3    | torque applied on the rotor between the front right two links     | -1           | 1            | angle_2 (front_right_leg)         | hinge       | torque (N m)  |
+------+-------------------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+---------------+
| 4    | torque applied on the rotor between the torso and back left hip   | -1           | 1            | hip_3 (back_leg)                  | hinge       | torque (N m)  |
+------+-------------------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+---------------+
| 5    | torque applied on the rotor between the back left two links       | -1           | 1            | angle_3 (back_leg)                | hinge       | torque (N m)  |
+------+-------------------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+---------------+
| 6    | torque applied on the rotor between the torso and back right hip  | -1           | 1            | hip_4 (right_back_leg)            | hinge       | torque (N m)  |
+------+-------------------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+---------------+
| 7    | torque applied on the rotor between the back right two links      | -1           | 1            | angle_4 (right_back_leg)          | hinge       | torque (N m)  |
+------+-------------------------------------------------------------------+--------------+--------------+-----------------------------------+-------------+---------------+


Specific Observation Space
--------------------------

+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| Size  | Observation                                                          | Min  | Max  | Name (in corresponding XML file)  | Joint/Site  | Unit                       |
+=======+======================================================================+======+======+===================================+=============+============================+
| 3     | accelerometer                                                        | -inf | inf  | accelerometer                     | site        | acceleration (m/s^2)       |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | velocimeter                                                          | -inf | inf  | velocimeter                       | site        | velocity (m/s)             |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | gyro                                                                 | -inf | inf  | gyro                              | site        | anglular velocity (rad/s)  |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | magnetometer                                                         | -inf | inf  | magnetometer                      | site        | magnetic flux (Wb)         |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 1     | angular velocity of angle between torso and front left link          | -Inf | Inf  | hip_1 (front_left_leg)            | hinge       | angle (rad)                |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 1     | angular velocity of the angle between front left links               | -Inf | Inf  | ankle_1 (front_left_leg)          | hinge       | angle (rad)                |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 1     | angular velocity of angle between torso and front right link         | -Inf | Inf  | hip_2 (front_right_leg)           | hinge       | angle (rad)                |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 1     | angular velocity of the angle between front right links              | -Inf | Inf  | ankle_2 (front_right_leg)         | hinge       | angle (rad)                |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 1     | angular velocity of angle between torso and back left link           | -Inf | Inf  | hip_3 (back_leg)                  | hinge       | angle (rad)                |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 1     | angular velocity of the angle between back left links                | -Inf | Inf  | ankle_3 (back_leg)                | hinge       | angle (rad)                |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 1     | angular velocity of angle between torso and back right link          | -Inf | Inf  | hip_4 (right_back_leg)            | hinge       | angle (rad)                |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 1     | angular velocity of the angle between back right links               | -Inf | Inf  | ankle_4 (right_back_leg)          | hinge       | angle (rad)                |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 1     | z-coordinate of the torso (centre)                                   | -Inf | Inf  | torso                             | site        | position (m)               |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 3     | xyz-coordinate angular velocity of the torso                         | -Inf | Inf  | torso                             | site        | angular velocity (rad/s)   |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 2     | sin() and cos() of angle between torso and first link on front left  | -Inf | Inf  | hip_1 (front_left_leg)            | hinge       | unitless                   |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 2     | sin() and cos() of angle between torso and first link on front left  | -Inf | Inf  | ankle_1 (front_left_leg)          | hinge       | unitless                   |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 2     | sin() and cos() of angle between torso and first link on front left  | -Inf | Inf  | hip_2 (front_right_leg)           | hinge       | unitless                   |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 2     | sin() and cos() of angle between torso and first link on front left  | -Inf | Inf  | ankle_2 (front_right_leg)         | hinge       | unitless                   |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 2     | sin() and cos() of angle between torso and first link on front left  | -Inf | Inf  | hip_3 (back_leg)                  | hinge       | unitless                   |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 2     | sin() and cos() of angle between torso and first link on front left  | -Inf | Inf  | ankle_3 (back_leg)                | hinge       | unitless                   |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 2     | sin() and cos() of angle between torso and first link on front left  | -Inf | Inf  | hip_4 (right_back_leg)            | hinge       | unitless                   |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+
| 2     | sin() and cos() of angle between torso and first link on front left  | -Inf | Inf  | ankle_4 (right_back_leg)          | hinge       | unitless                   |
+-------+----------------------------------------------------------------------+------+------+-----------------------------------+-------------+----------------------------+


Specific Starting Randomness
--------------------------------------------------


+-------------------+-------------------------------------+---------------+
| Scope             | Range                               | Distribution  |
+===================+=====================================+===============+
| angle of hip_1    | :math:`[0.5\pi-0.1, 0.5\pi+0.1]`    | uniform       |
+-------------------+-------------------------------------+---------------+
| angle of ankle_1  | :math:`[0.5\pi-0.1, 0.5\pi+0.1]`    | uniform       |
+-------------------+-------------------------------------+---------------+
| angle of hip_2    | :math:`[0.5\pi-0.1, 0.5\pi+0.1]`    | uniform       |
+-------------------+-------------------------------------+---------------+
| angle of ankle_2  | :math:`[-0.5\pi-0.1, -0.5\pi+0.1]`  | uniform       |
+-------------------+-------------------------------------+---------------+
| angle of hip_3    | :math:`[0.5\pi-0.1, 0.5\pi+0.1]`    | uniform       |
+-------------------+-------------------------------------+---------------+
| angle of ankle_3  | :math:`[-0.5\pi-0.1, -0.5\pi+0.1]`  | uniform       |
+-------------------+-------------------------------------+---------------+
| angle of hip_4    | :math:`[0.5\pi-0.1, 0.5\pi+0.1]`    | uniform       |
+-------------------+-------------------------------------+---------------+
| angle of ankle_4  | :math:`[0.5\pi-0.1, 0.5\pi+0.1]`    | uniform       |
+-------------------+-------------------------------------+---------------+

Specific Episode End
--------------------

- When Ant falls headfirst, the current episode ends: ``Terminated = True``.
