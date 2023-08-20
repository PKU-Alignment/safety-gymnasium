Safe Vision
===========

.. list-table::

    * - .. figure:: ../_static/images/building_goal0.jpeg
            :width: 230px
            :target: ../environments/safe_vision/building_goal.html#building_goal0
        .. centered:: :ref:`BuildingGoal0 <BuildingGoal0>`
      - .. figure:: ../_static/images/building_goal1.jpeg
            :width: 230px
            :target: ../environments/safe_vision/building_goal.html#building_goal1
        .. centered:: :ref:`BuildingGoal1 <BuildingGoal1>`
      - .. figure:: ../_static/images/building_goal2.jpeg
            :width: 230px
            :target: ../environments/safe_vision/building_goal.html#building_goal2
        .. centered:: :ref:`BuildingGoal2 <BuildingGoal2>`
    * - .. figure:: ../_static/images/building_button0.jpeg
            :width: 230px
            :target: ../environments/safe_vision/building_button.html#building_button0
        .. centered:: :ref:`BuildingButton0 <BuildingButton0>`
      - .. figure:: ../_static/images/building_button1.jpeg
            :width: 230px
            :target: ../environments/safe_vision/building_button.html#building_button1
        .. centered:: :ref:`BuildingButton1 <BuildingButton1>`
      - .. figure:: ../_static/images/building_button2.jpeg
            :width: 230px
            :target: ../environments/safe_vision/building_button.html#building_button2
        .. centered:: :ref:`BuildingButton2 <BuildingButton2>`
    * - .. figure:: ../_static/images/building_push0.jpeg
            :width: 230px
            :target: ../environments/safe_vision/building_push.html#building_push0
        .. centered:: :ref:`BuildingPush0 <BuildingPush0>`
      - .. figure:: ../_static/images/building_push1.jpeg
            :width: 230px
            :target: ../environments/safe_vision/building_push.html#building_push1
        .. centered:: :ref:`BuildingPush1 <BuildingPush1>`
      - .. figure:: ../_static/images/building_push2.jpeg
            :width: 230px
            :target: ../environments/safe_vision/building_push.html#building_push2
        .. centered:: :ref:`BuildingPush2 <BuildingPush2>`
    * - .. figure:: ../_static/images/fading_easy0.gif
            :width: 230px
            :target: ../environments/safe_vision/fading_easy.html#fading_easy0
        .. centered:: :ref:`FadingEasy0 <FadingEasy0>`
      - .. figure:: ../_static/images/fading_easy1.gif
            :width: 230px
            :target: ../environments/safe_vision/fading_easy.html#fading_easy1
        .. centered:: :ref:`FadingEasy1 <FadingEasy1>`
      - .. figure:: ../_static/images/fading_easy2.gif
            :width: 230px
            :target: ../environments/safe_vision/fading_easy.html#fading_easy2
        .. centered:: :ref:`FadingEasy2 <FadingEasy2>`
    * - .. figure:: ../_static/images/fading_hard0.gif
            :width: 230px
            :target: ../environments/safe_vision/fading_hard.html#fading_hard0
        .. centered:: :ref:`FadingHard0 <FadingHard0>`
      - .. figure:: ../_static/images/fading_hard1.gif
            :width: 230px
            :target: ../environments/safe_vision/fading_hard.html#fading_hard1
        .. centered:: :ref:`FadingHard1 <FadingHard1>`
      - .. figure:: ../_static/images/fading_hard2.gif
            :width: 230px
            :target: ../environments/safe_vision/fading_hard.html#fading_hard2
        .. centered:: :ref:`FadingHard2 <FadingHard2>`
    * - .. figure:: ../_static/images/race0.jpeg
            :width: 230px
            :target: ../environments/safe_vision/race.html#race0
        .. centered:: :ref:`Race0 <Race0>`
      - .. figure:: ../_static/images/race1.jpeg
            :width: 230px
            :target: ../environments/safe_vision/race.html#race1
        .. centered:: :ref:`Race1 <Race1>`
      - .. figure:: ../_static/images/race2.jpeg
            :width: 230px
            :target: ../environments/safe_vision/race.html#race2
        .. centered:: :ref:`Race2 <Race2>`
    * - .. figure:: ../_static/images/formula_one0.jpeg
            :width: 230px
            :target: ../environments/safe_vision/formula_one.html#formula_one0
        .. centered:: :ref:`FormulaOne0 <FormulaOne0>`
      - .. figure:: ../_static/images/formula_one1.jpeg
            :width: 230px
            :target: ../environments/safe_vision/formula_one.html#formula_one1
        .. centered:: :ref:`FormulaOne1 <FormulaOne1>`
      - .. figure:: ../_static/images/formula_one2.jpeg
            :width: 230px
            :target: ../environments/safe_vision/formula_one.html#formula_one2
        .. centered:: :ref:`FormulaOne2 <FormulaOne2>`

The Safe Vision tasks aim to promote research on robot safety in more intricate task settings, featuring tasks embedded with richer visual information. Objects within these environments are not only visually more complex but also present greater challenges in terms of nuanced collision intricacies.

.. Note::

    In tasks associated with Safe Vision, we have endeavored to support vectorized observation information, facilitating the community to glean new insights in more sophisticated environments.

    For all scenarios utilizing visual inputs, it's requisite to append the **Vision** suffix to the task name, as in ``SafetyAntGoal0Vision-v0``.

    Furthermore, even in the context of Safe Vision tasks, the absence of this suffix still implies vectorized observations. Hence, it can be inferred that all tasks within Safe Vision and Safe Navigation concurrently support both visual and vectorized observations. The distinctive categorization merely serves to emphasize the characteristics inherent to each task type.

.. toctree::
    :hidden:

    safe_vision/building_button.rst
    safe_vision/building_goal.rst
    safe_vision/building_push.rst
    safe_vision/fading_easy.rst
    safe_vision/fading_hard.rst
    safe_vision/race.rst
    safe_vision/formula_one.rst
