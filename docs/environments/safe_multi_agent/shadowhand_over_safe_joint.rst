.. _ShadowHandOverSafeJoint-MA:

ShadowHandOverSafeJoint(Multi-Agent)
====================================


.. list-table::
   :header-rows: 1

   * - Agent
   * - :doc:`../../components_of_environments/agents/shadowhands`


.. image:: ../../_static/images/shadow_hand_over_safe_joint.gif
    :align: center
    :scale: 26 %


This task is conceptualized from the ShadowHandOver as outlined in `Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learning <https://arxiv.org/abs/2206.08686>`__. Motivated by the real-world characteristics of ShadowHand, it integrates design constraints on the joints.

This scenario encompasses a specific environment comprising two Shadow Hands positioned opposite each other, with their palms facing upwards. The objective is to pass an object between these hands. Initially, the object will randomly descend within the area of the Shadow Hand on the right side. The hand on the right side then grasps the object and transfers it to the other hand. It is important to note that the base of each hand remains fixed throughout the process. Furthermore, the hand initially holding the object cannot directly make contact with the target hand or roll the object towards it. Hence, the object must be thrown into the air, maintaining its trajectory until it reaches the target hand.

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

+---------+----------------------------------+
| Index   | Description                      |
+=========+==================================+
| 0 - 19  | right Shadow Hand actuated joint |
+---------+----------------------------------+

Agent1
^^^^^^

+---------+----------------------------------+
| Index   | Description                      |
+=========+==================================+
| 0 - 19  | left Shadow Hand actuated joint  |
+---------+----------------------------------+

Rewards
-------

 Let the positions of the object and the goal be denoted as :math:`x_o` and :math:`x_g` respectively. The translational position difference between the object and the goal, represented as :math:`d_t`, can be computed as:

 .. math::

    d_t = \lVert x_o - x_g \rVert_2

 Similarly, we define the angular position difference between the object and the goal as :math:`d_a`. The rotational difference, denoted as :math:`d_r`, is then calculated as:

 .. math::

    d_r = 2 \arcsin(\mathrm{clamp}(\lVert d_a \rVert_2, \text{max} = 1.0))

 The rewards for the Hand Over task are determined using the following formula:

 .. math::

    r = \exp(-0.2(\alpha d_t + d_r))

 Here, :math:`\alpha` represents a constant that balances the rewards between translational and rotational aspects.



Costs
-----

.. list-table::

    * - .. figure:: ../../_static/images/shadow_hand_dof.jpg
            :scale: 20 %
      - .. figure:: ../../_static/images/shadow_hand_safe_joint.jpg
            :scale: 28 %

**Safety Joint** constrains the freedom of joint 4 of the forefinger. Without the constraint,
joint 4 has freedom of :math:`[-20^\circ,20^\circ]`. The safety tasks
restrict joint 4 within :math:`[-10^\circ, 10^\circ]`.
Let :math:`\mathtt{ang\_4}` be the angle of joint 4, and the cost is defined as:

.. math::

   c_t = \mathbb{I}(\mathtt{ang\_4} \not\in [-10^\circ, 10^\circ]).
