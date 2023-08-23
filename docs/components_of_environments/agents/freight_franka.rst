FreightFranka
=============



.. list-table::

    * - .. figure:: ../../_static/images/freight_franka_front.jpeg
            :width: 200px
        .. centered:: front
      - .. figure:: ../../_static/images/freight_franka_back.jpeg
            :width: 200px
        .. centered:: back
      - .. figure:: ../../_static/images/freight_franka_left.jpeg
            :width: 200px
        .. centered:: left
      - .. figure:: ../../_static/images/freight_franka_right.jpeg
            :width: 200px
        .. centered:: right


Franka Panda, designed by `Franka Emika <https://www.franka.de/>`__, serves as a versatile and user-friendly platform for robotics research and industrial tasks. The Franka Panda robot arm boasts 7 degrees of freedom, offering a high level of flexibility and dexterity. It can handle weights up to 3kg and has a reach of 855mm, making it suitable for a wide variety of tasks. The unique design integrates torque sensors in each of its joints, which allow for delicate operations, force control, and safe human-robot collaboration. The gripper at the end of the arm is equipped with force feedback capabilities, enabling precise grasping and manipulation of objects. Its ergonomic design, coupled with its powerful software platform, empowers users to deploy the robot arm in various applications, from research to industry, while ensuring safety and efficiency.

Fetch Freight, designed by `Fetch Robotics <https://www.fetchrobotics.com/>`__, is a cutting-edge mobile robot tailored for modern warehouses and distribution centers. With its autonomous navigation, it efficiently transports goods, navigating obstacles and ensuring safe delivery. Integrated with advanced sensors, the robot adapts to real-time changes and fits seamlessly into current warehouse systems. Its user-friendly interface and robust payload capacity make Fetch Freight an invaluable asset for enhancing logistics productivity and operations.

Combining Franka Panda with Fetch Freight creates a mobile manipulator platform, fusing precise dexterity with autonomous mobility. For robotics research, this integration allows for advanced experiments in complex environments, bridging the gap between manipulation and transportation tasks.



Specific Actions
----------------

+-----------+----------------------------------------------------------------------------------------------+
| Index     | Description                                                                                  |
+===========+==============================================================================================+
| 0         | x_joint of freight                                                                           |
+-----------+----------------------------------------------------------------------------------------------+
| 1         | y_joint of freight                                                                           |
+-----------+----------------------------------------------------------------------------------------------+
| 2         | z_rotation_joint of freight                                                                  |
+-----------+----------------------------------------------------------------------------------------------+
| 3         | panda_joint1                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 4         | panda_joint2                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 5         | panda_joint3                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 6         | panda_joint4                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 7         | panda_joint5                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 8         | panda_joint6                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 9         | panda_joint7                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 10        | panda_finger_joint1                                                                          |
+-----------+----------------------------------------------------------------------------------------------+
| 11        | panda_finger_joint2                                                                          |
+-----------+----------------------------------------------------------------------------------------------+


Specific Observations
---------------------

+-----------------+-------------------------------------------------------------------------------------------------------------+
| Index           | Description                                                                                                 |
+=================+=============================================================================================================+
| 0 - 9           | Joint DOF values                                                                                            |
+-----------------+-------------------------------------------------------------------------------------------------------------+
| 10 - 19         | Joint DOF velocities                                                                                        |
+-----------------+-------------------------------------------------------------------------------------------------------------+
| 20 - 22         | Relative pose between the Franka robot's root and the hand's rigid body tensor                              |
+-----------------+-------------------------------------------------------------------------------------------------------------+
| 23 - 32         | Actions taken by the robot in the joint space                                                               |
+-----------------+-------------------------------------------------------------------------------------------------------------+
