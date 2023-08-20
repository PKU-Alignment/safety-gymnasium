.. _ShadowHandCatchOver2UnderarmSafeFinger:

ShadowHandCatchOver2UnderarmSafeFinger
======================================


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

+-----------+----------------------------------------------------------------------------------------------+
| Index     | Description                                                                                  |
+===========+==============================================================================================+
| 0 - 397   | dual hands observation shown in ShadowHands agent section                                    |
+-----------+----------------------------------------------------------------------------------------------+
| 398 - 404 | object pose                                                                                  |
+-----------+----------------------------------------------------------------------------------------------+
| 405 - 407 | object linear velocity                                                                       |
+-----------+----------------------------------------------------------------------------------------------+
| 408 - 410 | object angle velocity                                                                        |
+-----------+----------------------------------------------------------------------------------------------+
| 411 - 417 | goal pose                                                                                    |
+-----------+----------------------------------------------------------------------------------------------+
| 418 - 421 | goal rot - object rot                                                                        |
+-----------+----------------------------------------------------------------------------------------------+


Actions
-------

+---------+------------------------------------+
| Index   | Description                        |
+=========+====================================+
| 0 - 19  | right Shadow Hand actuated joint   |
+---------+------------------------------------+
| 20 - 22 | right Shadow Hand base translation |
+---------+------------------------------------+
| 23 - 25 | right Shadow Hand base rotation    |
+---------+------------------------------------+
| 26 - 45 | left Shadow Hand actuated joint    |
+---------+------------------------------------+
| 46 - 48 | left Shadow Hand base translation  |
+---------+------------------------------------+
| 49 - 51 | left Shadow Hand base rotation     |
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
