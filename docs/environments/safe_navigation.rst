Safe Navigation
===============

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
    * - .. figure:: ../_static/images/button0.jpeg
            :width: 230px
            :target: safe_navigation/button.html#button0
        .. centered:: :ref:`Button0 <Button0>`
      - .. figure:: ../_static/images/button1.jpeg
            :width: 230px
            :target: safe_navigation/button.html#button1
        .. centered:: :ref:`Button1 <Button1>`
      - .. figure:: ../_static/images/button2.jpeg
            :width: 230px
            :target: safe_navigation/button.html#button2
        .. centered:: :ref:`Button2 <Button2>`
    * - .. figure:: ../_static/images/push0.jpeg
            :width: 230px
            :target: safe_navigation/push.html#push0
        .. centered:: :ref:`Push0 <Push0>`
      - .. figure:: ../_static/images/push1.jpeg
            :width: 230px
            :target: safe_navigation/push.html#push1
        .. centered:: :ref:`Push1 <Push1>`
      - .. figure:: ../_static/images/push2.jpeg
            :width: 230px
            :target: safe_navigation/push.html#push2
        .. centered:: :ref:`Push2 <Push2>`
    * - .. figure:: ../_static/images/circle0.jpeg
            :width: 230px
            :target: safe_navigation/circle.html#circle0
        .. centered:: :ref:`Circle0 <Circle0>`
      - .. figure:: ../_static/images/circle1.jpeg
            :width: 230px
            :target: safe_navigation/circle.html#circle1
        .. centered:: :ref:`Circle1 <Circle1>`
      - .. figure:: ../_static/images/circle2.jpeg
            :width: 230px
            :target: safe_navigation/circle.html#circle2
        .. centered:: :ref:`Circle2 <Circle2>`

Navigation tasks are an important class of tasks that apply RL to reality, requiring an agent to continuously change its position and interact with objects in the environment in order to accomplish a specified goal, which is usually associated with a specific position or movement pattern. In Safe RL, the focus is on the behavioral paradigm of whether an intelligent body, as a free-moving individual, can accomplish tasks in the environment without dangerous collisions or contact.

We have refactored and optimized the widely used but unmaintained and lacking supports environment library `Safety-Gym <https://github.com/openai/safety-gym>`__ in the library, and we have also carefully designed new environments and agents to take into account changing requirements, increasing computational power and advances in algorithms, which will be gradually Released in the coming period.


.. toctree::
    :hidden:

    safe_navigation/circle.rst
    safe_navigation/goal.rst
    safe_navigation/button.rst
    safe_navigation/push.rst
