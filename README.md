# Safety-Gymnasium
[![Organization](https://img.shields.io/badge/Organization-PKU_MARL-blue.svg)](https://github.com/PKU-MARL)[![PyPI](https://img.shields.io/pypi/v/safety-gymnasium?logo=pypi)](https://pypi.org/project/safety-gymnasium)[![Downloads](https://static.pepy.tech/personalized-badge/safety_gymnasium?period=total&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/safety_gymnasium)[![Documentation Status](https://img.shields.io/readthedocs/safety-gymnasium?logo=readthedocs)](https://safety-gymnasium.readthedocs.io)[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![License](https://img.shields.io/github/license/PKU-MARL/Safety-Gymnasium?label=license)](#license)

[**Why Safety-Gymnasium?**](#why-safety-gymnasium)| [**Environments**](#environments)| [**Installation**](#installation)| [**Documentation**](#documentation)| [**Design environments by yourself**](#design-environments-by-yourself)

**This library is currently under heavy development - if you have suggestions on the API or use-cases you'd like to be covered, please open an github issue or reach out. We'd love to hear about how you're using the library.**

Safety-Gymnasium is a highly scalable and customizable Safe Reinforcement Learning library, aiming to deliver a good view of benchmarking Safe Reinforcement Learning (Safe RL) algorithms and a more standardized setting of environments. We provide a set of standard API which is compatible with information on constraints. Users can explore new insights via an elegant code framework and well-designed environments.

--------------------------------------------------------------------------------

## Why Safety-Gymnasium?

Here we provide a table for comparison of **Safety-Gymnasium** and existing SafeRL Environments libraries.

|SafeRL<br/>Envs|Engine| Vectorized<br/> Environments | New Gym API<sup>**(3)**</sup> |    Vision Input     |
| :----------------------------------------------------------: | :---------------------------: | :-------------------: | :---------------------------: | :-----------------: |
| [Safety Gym](https://github.com/openai/safety-gym)<br/>![GitHub last commit](https://img.shields.io/github/last-commit/openai/safety-gym?label=last%20update) | `mujoco-py`<sup>**(1)**</sup>|  ❌  |               ❌               | minimally supported |
| [safe-control-gym](https://github.com/utiasDSL/safe-control-gym)<br/>![GitHub last commit](https://img.shields.io/github/last-commit/utiasDSL/safe-control-gym?label=last%20update)|           PyBullet           |         ❌               |               ❌               |          ❌          |
|            Velocity-Constraints<sup>**(2)**</sup>            |   N/A   |   ❌                 |         ❌          |               ❌               |          ❌          |
| [mujoco-circle](https://github.com/ymzhang01/mujoco-circle)<br/>![GitHub last commit](https://img.shields.io/github/last-commit/ymzhang01/mujoco-circle?label=last%20update) | PyTorch |  ❌|  ❌           |               ❌               |          ❌          |
| Safety Gymnaisum<br/>![GitHub last commit](https://img.shields.io/github/last-commit/PKU-MARL/Safety-Gymnasium) |      **MuJoCo 2.3.0+**        |   ✅  |               ✅               |          ✅          |

<sup>(1): Maintenance (expect bug fixes and minor updates); the last commit is 19 Nov 2021. Safety Gym depends on `mujoco-py` 2.0.2.7, which was updated on Oct 12, 2019.</sup><br/>
<sup>(2): There is no official library for speed-related environments, and its associated cost constraints are constructed from info. But the task is widely used in the study of SafeRL, and we encapsulate it in Safety-Gymnasium.</sup><br/>
<sup>(3): In the gym 0.26.0 release update, a new API of interaction was redefined.</sup>

--------------------------------------------------------------------------------

## Environments

We designed a variety of safety-enhanced learning tasks and integrated the contributions of RL community:`safety-velocity`, `safety-run`, `safety-circle`, `safety-goal`, `safety-button`, etc, leading to a unified safety-enhanced learning benchmark environment called `Safety-Gymnasium.`

Further, to facilitate the progress of community research, we redesigned [Safety Gym](https://github.com/openai/safety-gym) and removed the dependency on `mujoco-py`. We built it on top of [MuJoCo](https://github.com/deepmind/mujoco) and fixed some bugs, more specific bug report can refer to [Safety Gym's BUG Report](https://github.com/PKU-MARL/Safety-Gymnasium/blob/main/safety_gym_bug_report.md).

Here is a list of all the environments we support for now; some are being tested in our baselines, and we will gradually disclose it in the later updates.

<table border="1">
    <tr>
        <th>Category</th>
        <th>Task</th>
        <th>Agent</th>
    </tr>
    <tr>
        <td rowspan="4">Safe Navigation</td>
        <td>Goal[012]</td>
        <td rowspan="4">Point, Car, Racecar, Ant</td>
    </tr>
    <tr>
        <td>Button[012]</td>
    </tr>
    <tr>
        <td>Push[012]</td>
    </tr>
    <tr>
        <td>Circle[012]</td>
    </tr>
    <tr>
        <td>Safe Velocity</td>
        <td>Velocity</td>
        <td>HalfCheetah, Hopper, Swimmer, Walker2d, Ant, Humanoid</td>
    </tr>
</table>

Here is some pictures about tasks in Safe Navigation.

#### Agents

<table class="docutils align-default">
<tbody>
<tr class="row-odd"><td><figure class="align-default">
<a class="reference external image-reference"><img alt="./images/point_front.jpeg" src="./images/point_front.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="doc">Point</span></a></strong></p></td>
<td><figure class="align-default">
<a class="reference external image-reference" href="agents/car.html"><img alt="./images/car_front.jpeg" src="./images/car_front.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="doc">Car</span></a></strong></p></td>
<td><figure class="align-default">
<a class="reference external image-reference" href="agents/racecar.html"><img alt="./images/racecar_front.jpeg" src="./images/racecar_front.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="doc">Racecar</span></a></strong></p></td>
</tr>
<tr class="row-even"><td><figure class="align-default">
<a class="reference external image-reference" href="agents/ant.html"><img alt="./images/ant_front.jpeg" src="./images/ant_front.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="doc">Ant</span></a></strong></p></td>
<td><figure class="align-default">
<a class="reference internal image-reference"><img alt="./images/coming_soon.png" src="./images/coming_soon.png" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong>Coming soon…</strong></p></td>
<td><figure class="align-default">
<a class="reference internal image-reference"><img alt="./images/coming_soon.png" src="./images/coming_soon.png" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong>Coming soon…</strong></p></td>
</tr>
</tbody>
</table>

#### Tasks

<table class="docutils align-default">
<tbody>
<tr class="row-odd"><td><figure class="align-default">
<a class="reference external image-reference"><img alt="./images/goal0.jpeg" src="./images/goal0.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Goal0</span></a></strong></p></td>
<td><figure class="align-default">
<a class="reference external image-reference"><img alt="./images/goal1.jpeg" src="./images/goal1.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Goal1</span></a></strong></p></td>
<td><figure class="align-default">
<a class="reference external image-reference"><img alt="./images/goal2.jpeg" src="./images/goal2.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Goal2</span></a></strong></p></td>
</tr>
<tr class="row-even"><td><figure class="align-default">
<a class="reference external image-reference"><img alt="./images/button0.jpeg" src="./images/button0.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Button0</span></a></strong></p></td>
<td><figure class="align-default">
<a class="reference external image-reference" href="./button#button1"><img alt="./images/button1.jpeg" src="./images/button1.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Button1</span></a></strong></p></td>
<td><figure class="align-default">
<a class="reference external image-reference" href="./button#button2"><img alt="./images/button2.jpeg" src="./images/button2.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Button2</span></a></strong></p></td>
</tr>
<tr class="row-odd"><td><figure class="align-default">
<a class="reference external image-reference"><img alt="./images/push0.jpeg" src="./images/push0.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Push0</span></a></strong></p></td>
<td><figure class="align-default">
<a class="reference external image-reference"><img alt="./images/push1.jpeg" src="./images/push1.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Push1</span></a></strong></p></td>
<td><figure class="align-default">
<a class="reference external image-reference"><img alt="./images/push2.jpeg" src="./images/push2.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Push2</span></a></strong></p></td>
</tr>
<tr class="row-even"><td><figure class="align-default">
<a class="reference external image-reference"><img alt="./images/circle0.jpeg" src="./images/circle0.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Circle0</span></a></strong></p></td>
<td><figure class="align-default">
<a class="reference external image-reference"><img alt="./images/circle1.jpeg" src="./images/circle1.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Circle1</span></a></strong></p></td>
<td><figure class="align-default">
<a class="reference external image-reference" href="./circle#circle2"><img alt="./images/circle2.jpeg" src="./images/circle2.jpeg" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong><a class="reference internal"><span class="std std-ref">Circle2</span></a></strong></p></td>
</tr>
<tr class="row-odd"><td><figure class="align-default">
<a class="reference internal image-reference"><img alt="./images/coming_soon.png" src="./images/coming_soon.png" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong>Coming soon…</strong></p></td>
<td><figure class="align-default">
<a class="reference internal image-reference"><img alt="./images/coming_soon.png" src="./images/coming_soon.png" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong>Coming soon…</strong></p></td>
<td><figure class="align-default">
<a class="reference internal image-reference"><img alt="./images/coming_soon.png" src="./images/coming_soon.png" style="width: 230px;"></a>
</figure>
<p class="centered">
<strong>Coming soon…</strong></p></td>
</tr>
</tbody>
</table>

### Vision-base Safe RL

Vision-based safety reinforcement learning lacks realistic scenarios. Although the original `Safety Gym` could minimally support visual input, the scenarios were too homogeneous. To facilitate the validation of visual-based safety reinforcement learning algorithms, we have developed a set of realistic vision-based SafeRL tasks, which are currently being validated on the baseline. **In the later updates, we will release that part of the environment of `Safety-Gymnasium.`**

For the appetizer, the images are as follows:

<img src="./images/vision_input.png" width="100%"/>

### Environment Usage

**Notes:** We support explicitly express cost based on  [**Gymnasium APIs**](https://github.com/Farama-Foundation/Gymnasium).

```python
import safety_gymnasium

env_name = 'SafetyPointGoal1-v0'
env = safety_gymnasium.make(env_name)

obs, info = env.reset()
terminated = False

while not terminated:
    act = env.action_space.sample()
    obs, reward, cost, terminated, truncated, info = env.step(act)
    env.render()
```

--------------------------------------------------------------------------------

## Installation

### Install from PyPi
```
pip install safety-gymnasium Notes:coming as soon as possible.
```

### Install from source

```bash
conda create -n <virtual-env-name> python=3.8
conda activate <virtual-env-name>
git clone git@github.com:PKU-MARL/Safety-Gymnasium.git
cd Safety-Gymnasium
pip install -e .
```

--------------------------------------------------------------------------------

## Design environments by yourself

We construct a highly expandable framework of code so that you can easily comprehend it and design your own environments to facilitate your research with no more than 100 lines of code on average.

Here is a minimal example:

```python
# import the objects you want to use
# or you can define specific objects by yourself, just make sure obeying our specification
from safety_gymnasium.assets.geoms import Apples
from safety_gymnasium.bases import BaseTask

# inherit the basetask
class MytaskLevel0(BaseTask):
    def __init__(self, config):
        super().__init__(config=config)
		# define some properties
        self.num_steps = 500
        self.robot.placements = [(-0.8, -0.8, 0.8, 0.8)]
        self.robot.keepout = 0
        self.lidar_max_dist = 6
        # add objects into environments
        self.add_geoms(Apples(num=2, size=0.3))
        self.specific_agent_config()

    def calculate_reward(self):
        # implement your reward function
        # Note: cost calculation is based on objects, so it's automatic
        reward = 1
        return reward

    def specific_agent_config(self):
        # depending on your task
        pass

    def specific_reset(self):
        # depending on your task

    def specific_step(self):
        # depending on your task

    def build_goal(self):
        # depending on your task

    def update_world(self):
        # depending on your task

    @property
    def goal_achieved(self):
        # depending on your task
```

## License

Safety-Gymnasium is released under Apache License 2.0.
