.. _ShadowHandOverSafeFinger:

ShadowHandOverSafeFinger
========================


.. list-table::
   :header-rows: 1

   * - Agent
   * - :doc:`../../components_of_environments/agents/shadowhands`


.. image:: ../../_static/images/shadow_hand_over_safe_finger.gif
    :align: center
    :scale: 26 %

This task is modeled after the ShadowHandOver as proposed in `Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learning <https://arxiv.org/abs/2206.08686>`__. Drawing inspiration from the real-world features of ShadowHand, it introduces design constraints on the fingers.

This scenario encompasses a specific environment comprising two Shadow Hands positioned opposite each other, with their palms facing upwards. The objective is to pass an object between these hands. Initially, the object will randomly descend within the area of the Shadow Hand on the right side. The hand on the right side then grasps the object and transfers it to the other hand. It is important to note that the base of each hand remains fixed throughout the process. Furthermore, the hand initially holding the object cannot directly make contact with the target hand or roll the object towards it. Hence, the object must be thrown into the air, maintaining its trajectory until it reaches the target hand.

Observations
------------

+-----------+-----------------------------------------------------------------------------------------+
| Index     | Description                                                                             |
+===========+=========================================================================================+
| 0 - 373   | dual hands observation shown in ShadowHands agent section                               |
+-----------+-----------------------------------------------------------------------------------------+
| 374 - 380 | object pose                                                                             |
+-----------+-----------------------------------------------------------------------------------------+
| 381 - 383 | object linear velocity                                                                  |
+-----------+-----------------------------------------------------------------------------------------+
| 384 - 386 | object angle velocity                                                                   |
+-----------+-----------------------------------------------------------------------------------------+
| 387 - 393 | goal pose                                                                               |
+-----------+-----------------------------------------------------------------------------------------+
| 394 - 397 | goal rot - object rot                                                                   |
+-----------+-----------------------------------------------------------------------------------------+

Actions
-------

+---------+----------------------------------+
| Index   | Description                      |
+=========+==================================+
| 0 - 19  | right Shadow Hand actuated joint |
+---------+----------------------------------+
| 20 - 39 | left Shadow Hand actuated joint  |
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
