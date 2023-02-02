Safe Navigation
================

.. list-table:: 

    * - .. figure:: ../_static/images/goal0.jpeg
            :width: 230px
            :target: ./goal#goal0
        .. centered:: :ref:`Goal0 <Goal0>`
      - .. figure:: ../_static/images/goal1.jpeg
            :width: 230px
            :target: ./goal#goal1
        .. centered:: :ref:`Goal1 <Goal1>`
      - .. figure:: ../_static/images/goal2.jpeg
            :width: 230px
            :target: ./goal#goal2
        .. centered:: :ref:`Goal2 <Goal2>`
    * - .. figure:: ../_static/images/button0.jpeg
            :width: 230px
            :target: ./button#button0
        .. centered:: :ref:`Button0 <Button0>`
      - .. figure:: ../_static/images/button1.jpeg
            :width: 230px
            :target: ./button#button1
        .. centered:: :ref:`Button1 <Button1>`
      - .. figure:: ../_static/images/button2.jpeg
            :width: 230px
            :target: ./button#button2
        .. centered:: :ref:`Button2 <Button2>`
    * - .. figure:: ../_static/images/push0.jpeg
            :width: 230px
            :target: ./push#push0
        .. centered:: :ref:`Push0 <Push0>`
      - .. figure:: ../_static/images/push1.jpeg
            :width: 230px
            :target: ./push#push1
        .. centered:: :ref:`Push1 <Push1>`
      - .. figure:: ../_static/images/push2.jpeg
            :width: 230px
            :target: ./push#push2
        .. centered:: :ref:`Push2 <Push2>`
    * - .. figure:: ../_static/images/circle0.jpeg
            :width: 230px
            :target: ./circle#circle0
        .. centered:: :ref:`Circle0 <Circle0>`
      - .. figure:: ../_static/images/circle1.jpeg
            :width: 230px
            :target: ./circle#circle1
        .. centered:: :ref:`Circle1 <Circle1>`
      - .. figure:: ../_static/images/circle2.jpeg
            :width: 230px
            :target: ./circle#circle2
        .. centered:: :ref:`Circle2 <Circle2>`
    * - .. figure:: ../_static/images/coming_soon.png
            :width: 230px
        .. centered:: Coming soon...
      - .. figure:: ../_static/images/coming_soon.png
            :width: 230px
        .. centered:: Coming soon...
      - .. figure:: ../_static/images/coming_soon.png
            :width: 230px
        .. centered:: Coming soon...

Navigation任务是将RL应用到现实当中的一类重要任务，该类任务要求智能体在环境当中不断改变自身位置以及与环境当中物体交互，从而完成指定目标，其目标通常与特定的位置或运动方式相关联。在Safe RL当中是侧重于研究智能体作为自由行动的个体，能否实现在环境中完成任务的同时也不能发生危险的碰撞或接触的行为范式。

我们在库中对目前被广泛使用但欠缺维护和支持的环境库 `Safety-Gym <https://github.com/openai/safety-gym>`__ 做了重构和优化，同时考虑到需求的变化，计算能力的提高和算法的进步，我们还精心设计了新的环境和Agent，将会在今后的一段时间内逐步Release。


.. toctree::
    :hidden:

    safe_navigation/circle.rst
    safe_navigation/goal.rst
    safe_navigation/button.rst
    safe_navigation/push.rst


