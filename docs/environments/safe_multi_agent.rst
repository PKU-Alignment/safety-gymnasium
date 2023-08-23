Safe Multi-Agent
================

.. list-table::

    * - .. figure:: ../_static/images/ant_multi_goal0.jpeg
            :width: 230px
            :target: ../environments/safe_multi_agent/multi_goal.html#multi_goal0
        .. centered:: :ref:`MultiGoal0 <MultiGoal0>`
      - .. figure:: ../_static/images/ant_multi_goal1.jpeg
            :width: 230px
            :target: ../environments/safe_multi_agent/multi_goal.html#multi_goal1
        .. centered:: :ref:`MultiGoal1 <MultiGoal1>`
      - .. figure:: ../_static/images/ant_multi_goal2.jpeg
            :width: 230px
            :target: ../environments/safe_multi_agent/multi_goal.html#multi_goal2
        .. centered:: :ref:`MultiGoal2 <MultiGoal2>`


Furthermore, velocity constraints are extended to :ref:`multi-agent scenarios <MAVelocity>` while preserving the interface of MaMuJoCo, including the **following agents**:


.. Note::

    In MaMuJoCo, the same agent can be partitioned in various ways into multiple agents. We retained this feature, but established threshold values through experiments for only eight typical combinations.


.. list-table::

    * - .. figure:: ../_static/images/ant_vel.jpeg
            :width: 230px
            :target: ../environments/safe_multi_agent/velocity.html
        .. centered:: :ref:`MA-Ant <MAVelocity>`
      - .. figure:: ../_static/images/half_cheetah_vel.jpeg
            :width: 230px
            :target: ../environments/safe_multi_agent/velocity.html
        .. centered:: :ref:`MA-HalfCheetah <MAVelocity>`
      - .. figure:: ../_static/images/hopper_vel.jpeg
            :width: 230px
            :target: ../environments/safe_multi_agent/velocity.html
        .. centered:: :ref:`MA-Hopper <MAVelocity>`
    * - .. figure:: ../_static/images/humanoid_vel.jpeg
            :width: 230px
            :target: ../environments/safe_multi_agent/velocity.html
        .. centered:: :ref:`MA-Humanoid <MAVelocity>`
      - .. figure:: ../_static/images/swimmer_vel.jpeg
            :width: 230px
            :target: ../environments/safe_multi_agent/velocity.html
        .. centered:: :ref:`MA-Swimmer <MAVelocity>`
      - .. figure:: ../_static/images/walker2d_vel.jpeg
            :width: 230px
            :target: ../environments/safe_multi_agent/velocity.html
        .. centered:: :ref:`MA-Walker2d <MAVelocity>`


.. Note::

    **FreightFranka** presents a unique heterogeneous multi-agent scenario, drawing from instances in automated warehouses.


    The joint constraint limitations in **ShadowHands** strongly correlate with the challenges encountered in real-world settings. This is attributed to the fact that, although policies that perform well in simulation environments appear transferable to real-world scenarios, excessive control in practice can often result in significant damage.

.. tab-set::

    .. tab-item:: ShadowHand

        .. list-table::

                * - .. figure:: ../_static/images/shadow_hand_over_safe_finger.gif
                        :width: 350px
                        :target: ../environments/safe_multi_agent/shadowhand_over_safe_finger.html
                    .. centered:: :ref:`OverSafeFinger(Multi-Agent) <ShadowHandOverSafeFinger-MA>`
                  - .. figure:: ../_static/images/shadow_hand_catch_over2_underarm_safe_finger.gif
                        :width: 350px
                        :target: ../environments/safe_multi_agent/shadowhand_catch_over2_underarm_safe_finger.html
                    .. centered:: :ref:`CatchOver2UnderarmSafeFinger(Multi-Agent) <ShadowHandCatchOver2UnderarmSafeFinger-MA>`

        .. list-table::


                * - .. figure:: ../_static/images/shadow_hand_over_safe_joint.gif
                        :width: 350px
                        :target: ../environments/safe_multi_agent/shadowhand_over_safe_joint.html
                    .. centered:: :ref:`OverSafeJoint(Multi-Agent) <ShadowHandOverSafeJoint>`
                  - .. figure:: ../_static/images/shadow_hand_catch_over2_underarm_safe_joint.gif
                        :width: 350px
                        :target: ../environments/safe_multi_agent/shadowhand_catch_over2_underarm_safe_joint.html
                    .. centered:: :ref:`CatchOver2UnderarmSafeJoint(Multi-Agent) <ShadowHandCatchOver2UnderarmSafeJoint-MA>`

    .. tab-item:: FreightFranka

        .. list-table::

                * - .. figure:: ../_static/images/freight_franka_close_drawer.gif
                        :width: 350px
                        :target: ../environments/safe_multi_agent/freight_franka_close_drawer.html
                    .. centered:: :ref:`CloseDrawer(Multi-Agent) <FreightFrankaCloseDrawer-MA>`
                  - .. figure:: ../_static/images/freight_franka_pick_and_place.gif
                        :width: 350px
                        :target: ../environments/safe_multi_agent/freight_franka_pick_and_place.html
                    .. centered:: :ref:`PickAndPlace(Multi-Agent) <FreightFrankaPickAndPlace-MA>`

Safe Multi-Agent tasks extend certain original environments and agents to a multi-agent setting, enhancing the complexity of tasks and the interaction degrees of freedom among agents. The primary objective is to advance research on the safety aspects in multi-agent robotic scenarios.


.. toctree::
    :hidden:

    safe_multi_agent/multi_goal.rst
    safe_multi_agent/velocity.rst
    safe_multi_agent/freight_franka_close_drawer.rst
    safe_multi_agent/freight_franka_pick_and_place.rst
    safe_multi_agent/shadowhand_catch_over2_underarm_safe_finger.rst
    safe_multi_agent/shadowhand_catch_over2_underarm_safe_joint.rst
    safe_multi_agent/shadowhand_over_safe_finger.rst
    safe_multi_agent/shadowhand_over_safe_joint.rst
