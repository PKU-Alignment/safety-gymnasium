# Benchmark results in environments

We tested our environments on some safe reinforcement learning algorithms using [OmniSafe](https://github.com/PKU-Alignment/omnisafe) as an out-of-the-box RL framework. The results are shown below.

## Important notes
- All the algorithms are tested on the same hardware [AMD EPYC 7H12 64-Core Processor].
- The algorithms are tested with the same hyperparameters which are defaulted in [OmniSafe](https://github.com/PKU-Alignment/omnisafe/tree/main/omnisafe/configs/on-policy).
- Each algorithm is tested on at least 5 seeds and the average performance and standard deviation are shown in the figure. And the five seeds are [0, 5, 10, 15, 20]. We believe that the results are stable.
- The `Goal`, `Push`, and `Button` in `v0` series tasks of Safe Navigation are implemented as the original settings of [Safety-Gym](https://openai.com/research/safety-gym), We just made them more intelligible and accessible. The same situation happens on agents which appear in Safety-Gym. Further, we will fix some defects of design in the original Safety-Gym and release the new version of tasks as `v1` to facilitate research in the SafeRL community.
- The `Racecar` and `Ant` agents are newly added to `Safety-Gymnasium`, We are currently fine-tuning the physical parameters of these agents to make them more realistic and more compatible with various tasks. If you find any issues when using them, please feel free to open an issue, and we are welcome to PRs.

## Benchmark results
### SafetyAntVelocity-v1(1e6)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_ant_1e6.png">
    <br>
</center>

### SafetyAntVelocity-v1(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_ant_1e7.png">
    <br>
</center>

### SafetyHalfCheetahVelocity-v1(1e6)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_halfcheetah_1e6.png">
    <br>
</center>

### SafetyHalfCheetahVelocity-v1(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_halfcheetah_1e7.png">
    <br>
</center>

### SafetyHopperVelocity-v1(1e6)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_hopper_1e6.png">
    <br>
</center>

### SafetyHopperVelocity-v1(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_hopper_1e7.png">
    <br>
</center>

### SafetyHumanoidVelocity-v1(1e6)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_humanoid_1e6.png">
    <br>
</center>

### SafetyHumanoidVelocity-v1(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_humanoid_1e7.png">
    <br>
</center>

### SafetyWalker2dVelocity-v1(1e6)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_walker2d_1e6.png">
    <br>
</center>

### SafetyWalker2dVelocity-v1(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_walker2d_1e7.png">
    <br>
</center>

### SafetySwimmerVelocity-v1(1e6)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_swimmer_1e6.png">
    <br>
</center>

### SafetySwimmerVelocity-v1(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_swimmer_1e7.png">
    <br>
</center>

### SafetyPointGoal0-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointgoal0_1e7.png">
    <br>
</center>

### SafetyPointGoal1-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointgoal1_1e7.png">
    <br>
</center>

### SafetyPointGoal2-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointgoal2_1e7.png">
    <br>
</center>

### SafetyCarGoal0-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_cargoal0_1e7.png">
    <br>
</center>

### SafetyCarGoal1-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_cargoal1_1e7.png">
    <br>
</center>

### SafetyCarGoal2-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_cargoal2_1e7.png">
    <br>
</center>

### SafetyPointButton0-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointbutton0_1e7.png">
    <br>
</center>

### SafetyPointButton1-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointbutton1_1e7.png">
    <br>
</center>

### SafetyPointButton2-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointbutton2_1e7.png">
    <br>
</center>

### SafetyCarButton0-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_carbutton0_1e7.png">
    <br>
</center>

### SafetyCarButton1-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_carbutton1_1e7.png">
    <br>
</center>

### SafetyCarButton2-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_carbutton2_1e7.png">
    <br>
</center>

### SafetyPointPush0-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointpush0_1e7.png">
    <br>
</center>

### SafetyPointPush1-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointpush1_1e7.png">
    <br>
</center>

### SafetyPointPush2-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointpush2_1e7.png">
    <br>
</center>

### SafetyCarPush0-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_carpush0_1e7.png">
    <br>
</center>

### SafetyCarPush1-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_carpush1_1e7.png">
    <br>
</center>

### SafetyCarPush2-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_carpush2_1e7.png">
    <br>
</center>

### SafetyPointCircle0-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointcircle0_1e7.png">
    <br>
</center>

### SafetyPointCircle1-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointcircle1_1e7.png">
    <br>
</center>

### SafetyPointCircle2-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_pointcircle2_1e7.png">
    <br>
</center>

### SafetyCarCircle0-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_carcircle0_1e7.png">
    <br>
</center>

### SafetyCarCircle1-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_carcircle1_1e7.png">
    <br>
</center>

### SafetyCarCircle2-v0(1e7)
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_carcircle2_1e7.png">
    <br>
</center>
