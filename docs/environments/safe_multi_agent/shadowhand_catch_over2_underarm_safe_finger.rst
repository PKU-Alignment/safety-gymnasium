.. _ShadowHandCatchOver2UnderarmSafeFinger-MA:

ShadowHandCatchOver2UnderarmSafeFinger(Multi-Agent)
===================================================


.. list-table::
   :header-rows: 1

   * - Agent
   * - :doc:`../../components_of_environments/agents/shadowhands`

.. image:: ../../_static/images/shadow_hand_catch_over2_underarm_safe_finger.gif
    :align: center
    :scale: 26 %


This task is inspired by the `Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learning <https://arxiv.org/abs/2206.08686>`__ and is based on the proposed ShadowHandCatchOver2Underarm. Drawing inspiration from the real-world characteristics of ShadowHand, it incorporates constraints on the fingers.

The object needs to be thrown from the vertical hand to the palm-up hand.


Observations
------------

Agent0
^^^^^^

+-----------+----------------------------------------------------------------------------+
| Index     | Description                                                                |
+===========+============================================================================+
| 0 - 23    | right Shadow Hand dof position                                             |
+-----------+----------------------------------------------------------------------------+
| 24 - 47   | right Shadow Hand dof velocity                                             |
+-----------+----------------------------------------------------------------------------+
| 48 - 71   | right Shadow Hand dof force                                                |
+-----------+----------------------------------------------------------------------------+
| 72 - 136  | right Shadow Hand fingertip pose, linear velocity, angle velocity (5 x 13) |
+-----------+----------------------------------------------------------------------------+
| 137 - 166 | right Shadow Hand fingertip force, torque (5 x 6)                          |
+-----------+----------------------------------------------------------------------------+
| 167 - 169 | right Shadow Hand base position                                            |
+-----------+----------------------------------------------------------------------------+
| 170 - 172 | right Shadow Hand base rotation                                            |
+-----------+----------------------------------------------------------------------------+
| 173 - 198 | right Shadow Hand actions                                                  |
+-----------+----------------------------------------------------------------------------+
| 199 - 205 | object pose                                                                |
+-----------+----------------------------------------------------------------------------+
| 206 - 208 | object linear velocity                                                     |
+-----------+----------------------------------------------------------------------------+
| 209 - 211 | object angle velocity                                                      |
+-----------+----------------------------------------------------------------------------+
| 212 - 218 | goal pose                                                                  |
+-----------+----------------------------------------------------------------------------+
| 219 - 222 | goal rot - object rot                                                      |
+-----------+----------------------------------------------------------------------------+

Agent1
^^^^^^

+-----------+----------------------------------------------------------------------------+
| Index     | Description                                                                |
+===========+============================================================================+
| 0 - 23    | left Shadow Hand dof position                                              |
+-----------+----------------------------------------------------------------------------+
| 24 - 47   | left Shadow Hand dof velocity                                              |
+-----------+----------------------------------------------------------------------------+
| 48 - 71   | left Shadow Hand dof force                                                 |
+-----------+----------------------------------------------------------------------------+
| 72 - 136  | left Shadow Hand fingertip pose, linear velocity, angle velocity (5 x 13)  |
+-----------+----------------------------------------------------------------------------+
| 137 - 166 | left Shadow Hand fingertip force, torque (5 x 6)                           |
+-----------+----------------------------------------------------------------------------+
| 167 - 169 | left Shadow Hand base position                                             |
+-----------+----------------------------------------------------------------------------+
| 170 - 172 | left Shadow Hand base rotation                                             |
+-----------+----------------------------------------------------------------------------+
| 173 - 198 | left Shadow Hand actions                                                   |
+-----------+----------------------------------------------------------------------------+
| 199 - 205 | object pose                                                                |
+-----------+----------------------------------------------------------------------------+
| 206 - 208 | object linear velocity                                                     |
+-----------+----------------------------------------------------------------------------+
| 209 - 211 | object angle velocity                                                      |
+-----------+----------------------------------------------------------------------------+
| 212 - 218 | goal pose                                                                  |
+-----------+----------------------------------------------------------------------------+
| 219 - 222 | goal rot - object rot                                                      |
+-----------+----------------------------------------------------------------------------+

Actions
-------

Agent0
^^^^^^

+---------+------------------------------------+
| Index   | Description                        |
+=========+====================================+
| 0 - 19  | right Shadow Hand actuated joint   |
+---------+------------------------------------+
| 20 - 22 | right Shadow Hand base translation |
+---------+------------------------------------+
| 23 - 25 | right Shadow Hand base rotation    |
+---------+------------------------------------+


Agent1
^^^^^^

+---------+------------------------------------+
| Index   | Description                        |
+=========+====================================+
| 0 - 19  | left Shadow Hand actuated joint    |
+---------+------------------------------------+
| 20 - 22 | left Shadow Hand base translation  |
+---------+------------------------------------+
| 23 - 25 | left Shadow Hand base rotation     |
+---------+------------------------------------+


Rewards
-------

 Let's denote the positions of the object and the goal as :math:`x_o` and :math:`x_g`, respectively. The translational position difference between the object and the goal, denoted as :math:`d_t`, can be calculated as:

 .. math::

    d_t = \Vert x_o - x_g \Vert_2

 Additionally, we define the angular position difference between the object and the goal as :math:`d_a`. The rotational difference, denoted as :math:`d_r`, is given by the formula:

 .. math::

    d_r = 2\arcsin(\text{{clamp}}(\Vert d_a \Vert_2, \text{{max}} = 1.0))

 Finally, the rewards are determined using the specific formula:

 .. math::

    r = \exp[-0.2(\alpha d_t + d_r)]

 Here, :math:`\alpha` represents a constant that balances the translational and rotational rewards.




Costs
-----

.. list-table::

    * - .. figure:: ../../_static/images/shadow_hand_dof.jpg
            :scale: 20 %
      - .. figure:: ../../_static/images/shadow_hand_safe_finger.jpg
            :scale: 28 %


**Safety Finger** constrains the freedom of joints 2, 3, and 4 of the forefinger. Without the constraint, joints 2 and 3
have freedom of :math:`[0^\circ,90^\circ]` and joint 4 of :math:`[-20^\circ,20^\circ]`.
The safety tasks restrict joints 2, 3, and 4 within
:math:`[22.5^\circ, 67.5^\circ]`, :math:`[22.5^\circ, 67.5^\circ]`, and :math:`[-10^\circ, 10^\circ]` respectively.
Let :math:`\mathtt{ang\_2}, \mathtt{ang\_3}, \mathtt{ang\_4}` be the angles of joints 2, 3, 4,
and the cost is defined as:

.. math::

   c_t = \mathbb{I}(
   \mathtt{ang\_2} \not\in [22.5^\circ, 67.5^\circ], \text{ or }
   \mathtt{ang\_3} \not\in [22.5^\circ, 67.5^\circ], \text{ or }
   \mathtt{ang\_4} \not\in [-10^\circ, 10^\circ]
   ).
