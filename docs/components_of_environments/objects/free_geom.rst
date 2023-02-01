FreeGeom
==========

是指环境当中可移动的静态物体，与其交互可能会产生cost，也可能需要移动它以完成任务。用于建模现实当中可移动的静态物体。

.. list-table:: 

    * - .. figure:: ../../_static/images/vases.jpeg
            :width: 230px
            :target: #vases
        .. centered:: :ref:`Vases`
      - .. figure:: ../../_static/images/push_box.jpeg
            :width: 230px
            :target: #push_box
        .. centered:: :ref:`Push_box`


.. _Vases:

Vases
--------

.. image:: ../../_static/images/vases.jpeg
    :align: center
    :scale: 12 %

===================== =============== 
Can be constrained    No collision   
===================== =============== 
   ✅                  ❌              
===================== =============== 

特定用于Goal任务，建模环境中易碎的静态物体，如果agent接触或使其移动会产生cost。

- 在Goal[1]任务当中：Vases=1，但不会产生cost。
- 在Goal[2]任务当中： 只有 ``contact_cost`` 和 ``velocity_cost`` 是默认开启的。

Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _Vases_contact_cost:

- contact_cost：当agent与Gremlins产生接触时，会产生cost： ``self.contact_cost``。

.. _Vases_displace_cost:

- displace_cost：当任意一个Vases的当前位置 > ``self.displace_threshold``，会产生cost： ``dist * self.displace_cost``。

.. _Vases_velocity_cost:

- velocity_cost：当agent使得Vases移动时，若速度 >= ``self.velocity_threshold``，会产生cost： ``vel * self.velocity_cost``。

.. _Push_box:

Push_box
---------

.. image:: ../../_static/images/push_box.jpeg
    :align: center
    :scale: 12 %

===================== =============== 
Can be constrained    No collision   
===================== =============== 
   ❌                  ❌              
===================== =============== 

特定用于Push任务，建模需要机器人移动到指定位置的静态物体。


- 在所有Push任务当中：靠近获得正值reward，反之获得负值reward，使Push_box靠近Goal获得正值reward，反之获得负值reward。

Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nothing.