---
title: BaseTask
---

# BaseTask

## safety_gymnasium.bases.base_task

```{eval-rst}
.. autoclass:: safety_gymnasium.bases.base_task.BaseTask
```


###  DataClass

```{eval-rst}
.. autoclass:: safety_gymnasium.bases.base_task.LidarConf
.. autoclass:: safety_gymnasium.bases.base_task.RewardConf
.. autoclass:: safety_gymnasium.bases.base_task.CostConf
.. autoclass:: safety_gymnasium.bases.base_task.MechanismConf
.. autoclass:: safety_gymnasium.bases.base_task.ObservationInfo
```


### Methods


```{eval-rst}
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask.__init__
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask.dist_goal
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask.calculate_cost
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask.build_observation_space
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask._build_placements_dict
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask.toggle_observation_space
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask._build_world_config
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask._build_static_geoms_config
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask.build_goal_position
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask._placements_dict_from_object
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask.obs
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask._obs_lidar
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask._obs_lidar_natural
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask._obs_lidar_pseudo
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask._obs_compass
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask._obs_vision
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask._ego_xy
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask.calculate_reward
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask.specific_reset
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask.specific_step
.. autofunction:: safety_gymnasium.bases.base_task.BaseTask.update_world
```



### Additional Methods

```{eval-rst}
.. autoproperty:: safety_gymnasium.bases.base_task.BaseTask.goal_achieved
```
