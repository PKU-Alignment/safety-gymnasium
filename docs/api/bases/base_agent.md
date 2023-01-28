---
title: BaseAgent
---

# BaseAgent

## safety_gymnasium.bases.base_agent

```{eval-rst}
.. autoclass:: safety_gymnasium.bases.base_agent.BaseAgent
```


###  DataClass

```{eval-rst}
.. autoclass:: safety_gymnasium.bases.base_agent.SensorConf
.. autoclass:: safety_gymnasium.bases.base_agent.SensorInfo
.. autoclass:: safety_gymnasium.bases.base_agent.BodyInfo
.. autoclass:: safety_gymnasium.bases.base_agent.DebugInfo
```

### Methods

```{eval-rst}
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.__init__
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent._load_model
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent._init_body_info
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent._build_action_space
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent._init_jnt_sensors
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.set_engine
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.apply_action
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.build_sensor_observation_space
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.obs_sensor
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.get_sensor
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.dist_xy
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.world_xy
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.keyboard_control_callback
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.debug
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.is_alive
.. autofunction:: safety_gymnasium.bases.base_agent.BaseAgent.reset
```



### Additional Methods

```{eval-rst}
.. autoproperty:: safety_gymnasium.bases.base_agent.BaseAgent.com
.. autoproperty:: safety_gymnasium.bases.base_agent.BaseAgent.mat
.. autoproperty:: safety_gymnasium.bases.base_agent.BaseAgent.vel
.. autoproperty:: safety_gymnasium.bases.base_agent.BaseAgent.pos
```
