# Car

一个稍微复杂一些的机器人，在三维空间中运动，它有两个独立驱动的平行车轮和一个自由滚动的后轮。对于这个机器人来说，无论是转向还是向前/向后移动都需要协调两个驱动器。它在设计上类似于用于教育的简单机器人。

| Specific Action Space      | Box(-1.0, 1.0, (2,), float64) |
| -------------------------- | ----------------------------- |
| Specific Observation Shape | (24,)                         |
| Observation High           | inf                           |
| Observation Low            | -inf                          |

## Specific Action Space

| Num  | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint/Site | Unit      |
| ---- | ------------------------- | ----------- | ----------- | -------------------------------- | ---------- | --------- |
| 0    | To applied on left wheel  | 0           | 1           | left                             | hinge      | Force (N) |
| 1    | To applied on right wheel | 0           | 1           | right                            | hinge      | Force (N) |

## Specific Observation Space

| Size | Observation                                                  | Min  | Max  | Name (in corresponding XML file) | Joint/Site | Unit                      |
| ---- | ------------------------------------------------------------ | ---- | ---- | -------------------------------- | ---------- | ------------------------- |
| 9    | Quaternions of the rear wheel which are turned into 3x3 rotation matrices. | -inf | inf  | ballquat_rear                    | ball       | unitless                  |
| 3    | Angle velocity of the rear wheel.                            | -inf | inf  | ballangvel_rear                  | ball       | anglular velocity (rad/s) |
| 3    | accelerometer                                                | -inf | inf  | accelerometer                    | site       | acceleration (m/s^2)      |
| 3    | velocimeter                                                  | -inf | inf  | velocimeter                      | site       | velocity (m/s)            |
| 3    | gyro                                                         | -inf | inf  | gyro                             | site       | anglular velocity (rad/s) |
| 3    | magnetometer                                                 | -inf | inf  | magnetometer                     | site       | magnetic flux (Wb)        |

## Specific Starting Randomness

None