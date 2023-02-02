简介
====


关于Safe RL
-----------

In recent years, RL (reinforcement learning) algorithms, especially DeepRL algorithms, have achieved good performance in numerous tasks. For example: gaining high scores in Atari games using only visual input, fulfilling complex control tasks in high dimensions, and beating human grandmasters in Go competitions. However, in the process of policy optimization in RL, Agents **often** learn **reward hacking** or even **risky behavior** to **improve their cumulative rewards**. This may **lead to one-sided pursuit of reward** and **contradict the original purpose** of our reward design. Therefore, the **Safe RL algorithm aims to ** train the Agent to ** maximize reward while satisfying the given constraints**, so as to avoid learning behavioral policies that are out of touch with reality and one-sidedly pursue reward.

强化学习可以理解为Agent通过给定的reward信号学习，不断优化自身策略，在解决无法严格进行数学建模的问题时十分有效。在此基础上，Safe RL也可以广义地理解为一个**约束求解问题**：Agent需要在不断学习reward信号的同时对约束(cost)信号进行学习，从而对难以有效建模的约束进行学习，在满足约束的前提下，最大化reward。

Safety-Gymnasium
----------------

Safety-Gymnasium是为了服务于Safe RL领域研究而基于Python开发的一个 **高度模块化**，**代码精简易读** 并且 **易于自定义** 的基准环境。

Feature
^^^^^^^^

- 高度模块化，将一个环境的不同组成部分拆分为逻辑自洽的不同模块：agent, task, objects。
- 代码精简易读，在保证 **可读性** 和 **美观性** 的前提下最大程度精简代码，每一个任务平均 **不超过100行** 代码。
- 易于自定义，**精心设计的代码框架** 对于自定义环境的需求十分友好。
- 丰富可靠的环境，提供了 **Manipualtion** 和 **Vision** 两个新类别的任务。
- 对经典Safe RL环境的良好支持与重构： **Safety-Gym** ，**Safety-Velocity**。
- 依赖少。
- 易于安装。

