Mocap
==========

是指环境当中按照一定规律自主移动的物体，与其交互可能会产生cost，也可以通过物理交互影响其运动方式。用于建模现实中受控制的运动物体。

.. image:: ../../_static/images/mocap.png
    :align: center

Gremlins(can be constrained)
-------------------------

特定用于Button任务，建模环境当中移动的物体。

- 在Button[12]当中：与其接触会产生cost。

Cost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- contact_cost：当agent与Gremlins产生接触时，会产生cost： ``self.contact_cost``。

