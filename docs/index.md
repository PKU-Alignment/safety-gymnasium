---
hide-toc: true
firstpage:
lastpage:
---

# Safety Gymnasium is a standard API for safe reinforcement learning, and a diverse collection of reference environments

```{figure} _static/images/racecar_demo.gif
   :alt: racecar
   :width: 200
```

```{code-block} python

import safety_gymnasium
env = safety_gymnasium.make("Safety_RacecarGoal1-v0", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
```


```{toctree}
:hidden:
:caption: API

api/builder
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/PKU-MARL/Safety-Gymnasium>
Contribute to the Docs <https://github.com/PKU-MARL/Safety-Gymnasium/blob/main/CONTRIBUTING.md>
```
