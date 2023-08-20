Safe Isaac Gym
==============

Safe Isaac Gym introduces constraints based on real-world requirements and designing the heterogeneous multi-agent system, FreightFranka.

.. Note::

    By harnessing the rapid parallel capabilities of Isaac Gym, we are able to explore more realistic and challenging environments, unveiling and examining the potentialities of SafeRL. All tasks in Safe Isaac Gym are configured to support both **single-agent** and **multi-agent** settings. The single-agent and multi-agent algorithms from `SafePO <https://github.com/PKU-Alignment/Safe-Policy-Optimization>`__ can be seamlessly implemented in these respective environments.


.. list-table::

    * - .. figure:: ../_static/images/freight_franka_close_drawer.gif
            :width: 350px
            :target: ../environments/safe_isaac_gym/freight_franka_close_drawer.html
        .. centered:: :ref:`FreightFrankaCloseDrawer <FreightFrankaCloseDrawer>`
      - .. figure:: ../_static/images/freight_franka_pick_and_place.gif
            :width: 350px
            :target: ../environments/safe_isaac_gym/freight_franka_pick_and_place.html
        .. centered:: :ref:`FreightFrankaPickAndPlace <FreightFrankaPickAndPlace>`
    * - .. figure:: ../_static/images/shadow_hand_over_safe_finger.gif
            :width: 350px
            :target: ../environments/safe_isaac_gym/shadowhand_over_safe_finger.html
        .. centered:: :ref:`ShadowHandOverSafeFinger <ShadowHandOverSafeFinger>`
      - .. figure:: ../_static/images/shadow_hand_catch_over2_underarm_safe_finger.gif
            :width: 350px
            :target: ../environments/safe_isaac_gym/shadowhand_catch_over2_underarm_safe_finger.html
        .. centered:: :ref:`ShadowHandCatchOver2UnderarmSafeFinger <ShadowHandCatchOver2UnderarmSafeFinger>`
    * - .. figure:: ../_static/images/shadow_hand_over_safe_joint.gif
            :width: 350px
            :target: ../environments/safe_isaac_gym/shadowhand_over_safe_joint.html
        .. centered:: :ref:`ShadowHandOverSafeJoint <ShadowHandOverSafeJoint>`
      - .. figure:: ../_static/images/shadow_hand_catch_over2_underarm_safe_joint.gif
            :width: 350px
            :target: ../environments/safe_isaac_gym/shadowhand_catch_over2_underarm_safe_joint.html
        .. centered:: :ref:`ShadowHandCatchOver2UnderarmSafeJoint <ShadowHandCatchOver2UnderarmSafeJoint>`


.. toctree::
    :hidden:

    safe_isaac_gym/freight_franka_close_drawer.rst
    safe_isaac_gym/freight_franka_pick_and_place.rst
    safe_isaac_gym/shadowhand_catch_over2_underarm_safe_finger.rst
    safe_isaac_gym/shadowhand_catch_over2_underarm_safe_joint.rst
    safe_isaac_gym/shadowhand_over_safe_finger.rst
    safe_isaac_gym/shadowhand_over_safe_joint.rst
