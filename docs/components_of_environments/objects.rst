Objects
=======

.. list-table:: 

    * - .. figure:: ../_static/images/goal.jpeg
            :width: 230px
            :target: objects/geom.html#goal
        .. centered:: :ref:`Goal`
      - .. figure:: ../_static/images/buttons.jpeg
            :width: 230px
            :target: objects/geom.html#buttons
        .. centered:: :ref:`Buttons`
      - .. figure:: ../_static/images/circle.jpeg
            :width: 230px
            :target: objects/geom.html#circle
        .. centered:: :ref:`Circle`
    * - .. figure:: ../_static/images/pillars.jpeg
            :width: 230px
            :target: objects/geom.html#pillars
        .. centered:: :ref:`Pillars`
      - .. figure:: ../_static/images/hazards.jpeg
            :width: 230px
            :target: objects/geom.html#hazards
        .. centered:: :ref:`Hazards`
      - .. figure:: ../_static/images/sigwalls.jpeg
            :width: 230px
            :target: objects/geom.html#sigwalls
        .. centered:: :ref:`Sigwalls`
    * - .. figure:: ../_static/images/vases.jpeg
            :width: 230px
            :target: objects/free_geom.html#vases
        .. centered:: :ref:`Vases`
      - .. figure:: ../_static/images/push_box.jpeg
            :width: 230px
            :target: objects/free_geom.html#push-box
        .. centered:: :ref:`Push_box`
      - .. figure:: ../_static/images/gremlins.jpeg
            :width: 230px
            :target: objects/mocap.html#gremlins
        .. centered:: :ref:`Gremlins`

环境当中的物体分为3类： **Geom**, **FreeGeom**, **Mocap**。

- Geom：是指环境当中不可通过接触和碰撞来改变位置的静态物体。用于建模现实当中固定的静态物体。
- FreeGeom：是指环境当中可移动的静态物体，与其交互可能会产生cost，也可能需要移动它以完成任务。用于建模现实当中可移动的静态物体。
- Mocap：是指环境当中按照一定规律自主移动的物体，与其交互可能会产生cost，也可以通过物理交互影响其运动方式。用于建模现实中受控制的运动物体。

.. Note::

    1. 环境当中的物体有的有且仅有一个实例，而有的可以有多个，这一点我们通过命名的单复数形式来区分，例如：Vases，意味着这个物体可以有多个实例，而Goal意味着这个物体有且仅有一个实例。

    2. 环境当中的物体性质有差异，比如：有的物体可以参与计算cost，有的没有碰撞实体。下文介绍当中我们将给出提示。
    3. 所有可参与计算cost的物体根据任务难度可能不会成为约束，例如：Goal1当中碰撞vases并不产生cost。
    4. 你可以根据需要定义或改变物体的 **cost计算公式** ，**数量** ，**位置** ，**碰撞属性** ，**密度** ，**移动范式** 等，以探索不同情况下RL算法的表现。

通用参数
--------

每一个物体都有与环境交互需要的 **自定义参数** 和 **方法**。

- 通过改变已有物体这些参数的值，可以改变环境的行为，以个性化测试算法。

- 通过在我们提供的协议下定义 **新的** 一套 **参数** 和 **交互方式**，可以 **实现理想的物体特性**。这个过程包括 **参数的定义** 和 **方法的实现**。

.. code-block:: python

    @dataclass
    class Example(Geoms):

        name: str  # 物体的名字，是类名的小写。
        num: int  # 物体的数量。Note：若为唯一的物体，则没有该属性
        size: float = 0.3  # 物体的尺寸，视具体的形状而言，可能由多个可自定义的成员变量来共同决定。
        # 物体位置随机采样的区域，可以指定多个区域，从中均匀随机采样。
        # 每一个区域的格式为(xmin, ymin, xmax, ymax)，以list包裹
        placements: list

        # 仅能填写二维坐标
        # 显式指定前i个该物体的位置，i为填写的xy坐标数量
        locations: list
        # 采样坐标时判断是否与其他物体位置冲突预留的距离
        # 一般设定为与物体半径一样大
        keepout: float = 0.3  # Keepout radius when placing goals
        
        # 在Simulator当中显示的颜色
        color: np.array = COLOR['apple']
        # 划分group，服务于某些机制
        # 例如：lidar
        group: np.array = GROUP['apple']
        # 在当前环境中是否被雷达观测
        is_lidar_observed: bool = True
        # 在当前环境中是否被指南针观测，仅支持数量恒为1的物体。
        is_comp_observed: bool = False
        # 在当前环境中是否参与约束。
        is_constrained: bool = False


雷达机制
--------

在我们的库当中，通过雷达向agent提供关于物体的观测信息。

.. Note::
    这也意味着，Safe Navigation类别当中的任务，所有的观测都是本地信息，不包含全局的环境信息，我们认为这更贴近于现实当中机器人所能得到的观测。

Natural lidar
^^^^^^^^^^^^^

激光雷达通过Mujoco提供的接口实现，从机制上对应于现实当中的激光雷达。

.. Note::
    Natural lidar类别的每一个雷达的返回值是当前探测到的物体距离激光发射点的距离，若没有探测到，则为-1。

Pseudo lidar
^^^^^^^^^^^^

伪激光雷达的工作原理是循环场景中的所有的该类别物体，确定其是否在范围内，然后填充对应位置的雷达观测值。

两种雷达都是针对某一个类别的特定目标来设计的，并且会忽略其他类别的目标。例如：Vases lidar只能检测Vases，而Goal lidar只能检测Goal。

.. Note::
    在task的lidar_conf数据类当中，若指定了 :attr:`max_dist`,那么Pseudo lidar会按照如下公式给出observation：

    .. math:: O_i = \frac{D_i}{D_{max}}

    其中 :math:`O_i` 表示第i个雷达的数值， :math:`D_i` 表示第i个雷达的距离， :math:`D_{max}` 表示雷达的最大探测距离。

    否则，将按照指数衰减的方式来给出数值：

    .. math:: O_i = e^{-\alpha D_i}

    其中 :math:`\alpha` 是衰减系数。

.. hint::
    在task的lidar_conf数据类当中，通过修改lidar_type，可以切换雷达类别，但Natural lidar会显著更难。

Group mechanism
^^^^^^^^^^^^^^^

我们在底层实现当中，将不同的物体分为不同的group，每一个group通过不同的雷达来探测。
举例来说：对于探测Goal的雷达而言，Buttons是不存在的，反之亦然。

.. Note:: 
    这样的机制在可视化时也会被体现出来：

    .. list-table::

        * - .. figure:: ../_static/images/group_lidar_left.jpeg
                :width: 230px
          - .. figure:: ../_static/images/group_lidar_right.jpeg
                :width: 230px

    可以看到，当物体种类变多时，呈现相应数量的互不干扰的雷达。

Render lidar
^^^^^^^^^^^^

雷达在render时会被可视化，对应于agent头顶上的围成圆形的小球，小球亮起的颜色与物体颜色一致。当一个位置的雷达探测到目标时，会亮起，越接近目标，颜色越深。

以下是与物体距离不同情况下雷达的可视化效果：

.. list-table::

    * - .. figure:: ../_static/images/lidar_far.jpeg
            :width: 230px
        .. centered:: far
      - .. figure:: ../_static/images/lidar_mid.jpeg
            :width: 230px
        .. centered:: mid
      - .. figure:: ../_static/images/lidar_near.jpeg
            :width: 230px
        .. centered:: near

.. Note::
    雷达可视化标记是没有实体的，仅仅服务于人类观众。



.. toctree::
    :hidden:

    objects/geom.rst
    objects/free_geom.rst
    objects/mocap.rst


