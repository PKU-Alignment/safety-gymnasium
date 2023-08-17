Safe Vision
===========

.. list-table::

    * - .. figure:: ../_static/images/goal0.jpeg
            :width: 230px
            :target: safe_navigation/goal.html#goal0
        .. centered:: :ref:`Goal0 <Goal0>`
      - .. figure:: ../_static/images/goal1.jpeg
            :width: 230px
            :target: safe_navigation/goal.html#goal1
        .. centered:: :ref:`Goal1 <Goal1>`
      - .. figure:: ../_static/images/goal2.jpeg
            :width: 230px
            :target: safe_navigation/goal.html#goal2
        .. centered:: :ref:`Goal2 <Goal2>`
    * - .. figure:: ../_static/images/coming_soon.png
            :width: 230px
        .. centered:: Coming soon...
      - .. figure:: ../_static/images/coming_soon.png
            :width: 230px
        .. centered:: Coming soon...
      - .. figure:: ../_static/images/coming_soon.png
            :width: 230px
        .. centered:: Coming soon...

Safe vision tasks旨在通过一系列具备更真实视觉信息的任务，在更复杂的任务设定上促进机器人安全性的研究，此类环境中的物体
不仅仅是在视觉信息上更复杂，同时也在环境的碰撞细微程度上更具挑战性。

.. Note::

    在safe vision的任务当中，也尽可能支持了vectorized的观测信息，以便于社区在更复杂的环境当中探索新的见解。

    对于所有采用视觉输入的情形，需要在任务名称后面加上 **Vision** 后缀，例如 `SafetyAntGoal0Vision-v0` 。

    而即使是safe vision的任务，在不加该后缀的情况下也是vectorized观测的。因此可以认为，Safety-Gymnasium当中的所有任务都是同时支持视觉和vectorized观测的，不同的类别划分只是为了强调每一类任务的特点。

.. toctree::
    :hidden:

    safe_vision/building_button.rst
    safe_vision/building_goal.rst
    safe_vision/building_push.rst
    safe_vision/fading_easy.rst
    safe_vision/fading_hard.rst
    safe_vision/race.rst
    safe_vision/formula_one.rst
