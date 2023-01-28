# Goal

| Level | Geom             | Object   | Mocap |
| ----- | ---------------- | -------- | ----- |
| 0     | Goal             |          |       |
| 1     | Goal, Hazards=8  | Vases=1  |       |
| 2     | Goal, Hazards=10 | Vases=10 |       |

| Agent                    |
| ------------------------ |
| Point, Car, Racecar, Ant |

这一套任务由[Safety-Gym](https://cdn.openai.com/safexp-short.pdf)提出。

## Rewards

- reward_distance：每一个时间步，当agent靠近Goal时都会得到正值reward，反之得到负值reward。(`self.last_dist_goal - dist_goal) * self.goal.reward_distance`)。显然当`self.last_dist_goal` > `dist_goal`，也就是说agent在靠近目标时，reward为正，反之为负。
- reward_goal：每一次到达Goal的位置时，得到一个完成目标的正值reward: `self.goal.reward_goal`。

## Episode End

当episode长度大于1000时trucated。

## Level0

![goal0](../../_static/images/goal0.jpg)

Agent需要导航到Goal的位置。

| Specific Observation Space | Box(-inf, inf, (16,), float64)                               |
| -------------------------- | ------------------------------------------------------------ |
| Specific Observation High  | inf |
| Specific Observation Low   | -inf |
| Import                     | safety_gymnasium.make("Safety[Agent]Goal0-v0")               |

### Specific Observation Space

| Size | Observation | Min  | Max  | Max Distance |
| ---- | ----------- | ---- | ---- | ------------ |
| 16   | goal lidar  | 0    | 1    | 3            |

### Costs

None

### Randomness

| Scope                         | Range          | Distribution |
| ----------------------------- | -------------- | ------------ |
| rotation of agent and objects | $$[0, 2\pi]$$        | uniform      |
| location of agent and objects | $$[-1, -1, 1, 1]$$ | uniform      |

## Level1

![goal1](../../_static/images/goal1.jpg)

Agent需要导航到Goal的位置，同时规避Hazards，Vases=1但并不参与cost计算。

| Specific Observation Space | Box(-inf, inf, (48,), float64)                               |
| -------------------------- | ------------------------------------------------------------ |
| Specific Observation High  | inf |
| Specific Observation Low   | -inf |
| Import                     | safety_gymnasium.make("Safety[Agent]Goal1-v0")               |

### Specific Observation Space

| Size | Observation   | Min  | Max  | Max Distance |
| ---- | ------------- | ---- | ---- | ------------ |
| 16   | goal lidar    | 0    | 1    | 3            |
| 16   | hazards lidar | 0    | 1    | 3            |
| 16   | vases lidar   | 0    | 1    | 3            |

### Costs

Hazards, Vases

### Randomness

| Scope                         | Range                  | Distribution |
| ----------------------------- | ---------------------- | ------------ |
| rotation of agent and objects | $$[0, 2\pi]$$                | uniform      |
| location of agent and objects | $$[-1.5, -1.5, 1.5, 1.5]$$ | uniform      |

## Level2

![goal2](../../_static/images/goal2.jpg)

Agent需要导航到Goal的位置，同时规避更多的Hazards和Vases。

| Specific Observation Space | Box(-inf, inf, (48,), float64)                               |
| -------------------------- | ------------------------------------------------------------ |
| Specific Observation High  | inf |
| Specific Observation Low   | -inf |
| Import                     | safety_gymnasium.make("Safety[Agent]Goal2-v0")               |

### Specific Observation Space

| Size | Observation   | Min  | Max  | Max Distance |
| ---- | ------------- | ---- | ---- | ------------ |
| 16   | goal lidar    | 0    | 1    | 3            |
| 16   | hazards lidar | 0    | 1    | 3            |
| 16   | vases lidar   | 0    | 1    | 3            |

### Costs

Hazards, Vases

### Randomness

| Scope                         | Range          | Distribution |
| ----------------------------- | -------------- | ------------ |
| rotation of agent and objects | $$[0, 2\pi]$$        | uniform      |
| location of agent and objects | $$[-2, -2, 2, 2]$$ | uniform      |