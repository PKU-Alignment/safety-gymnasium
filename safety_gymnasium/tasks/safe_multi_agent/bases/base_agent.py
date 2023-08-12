# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class for agents."""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field

import glfw
import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

import safety_gymnasium
from safety_gymnasium.tasks.safe_multi_agent.utils.random_generator import RandomGenerator
from safety_gymnasium.tasks.safe_multi_agent.utils.task_utils import get_body_xvelp, quat2mat
from safety_gymnasium.tasks.safe_multi_agent.world import Engine


BASE_DIR = os.path.join(os.path.dirname(safety_gymnasium.__file__), 'tasks/safe_multi_agent')


@dataclass
class SensorConf:
    r"""Sensor observations configuration.

    Attributes:
        sensors (tuple): Specify which sensors to add to observation space.
        sensors_hinge_joints (bool): Observe named joint position / velocity sensors.
        sensors_ball_joints (bool): Observe named ball joint position / velocity sensors.
        sensors_angle_components (bool): Observe sin/cos theta instead of theta.
    """

    sensors: tuple = (
        'accelerometer',
        'velocimeter',
        'gyro',
        'magnetometer',
        'accelerometer1',
        'velocimeter1',
        'gyro1',
        'magnetometer1',
    )
    sensors_hinge_joints: bool = True
    sensors_ball_joints: bool = True
    sensors_angle_components: bool = True


@dataclass
class SensorInfo:
    r"""Sensor information generated in running.

    Needed to figure out observation space.

    Attributes:
        hinge_pos_names (list): List of hinge joint position sensor names.
        hinge_vel_names (list): List of hinge joint velocity sensor names.
        freejoint_pos_name (str): Name of free joint position sensor.
        freejoint_qvel_name (str): Name of free joint velocity sensor.
        ballquat_names (list): List of ball joint quaternion sensor names.
        ballangvel_names (list): List of ball joint angular velocity sensor names.
        sensor_dim (list): List of sensor dimensions.
    """

    hinge_pos_names: list = field(default_factory=list)
    hinge_vel_names: list = field(default_factory=list)
    freejoint_pos_name: str = None
    freejoint_qvel_name: str = None
    ballquat_names: list = field(default_factory=list)
    ballangvel_names: list = field(default_factory=list)
    sensor_dim: list = field(default_factory=dict)


@dataclass
class BodyInfo:
    r"""Body information generated in running.

    Needed to figure out the observation spaces.

    Attributes:
        nq (int): Number of generalized coordinates in agent = dim(qpos).
        nv (int): Number of degrees of freedom in agent = dim(qvel).
        nu (int): Number of actuators/controls in agent = dim(ctrl),
            needed to figure out action space.
        nbody (int): Number of bodies in agent.
        geom_names (list): List of geom names in agent.
    """

    nq: int = None
    nv: int = None
    nu: int = None
    nbody: int = None
    geom_names: list = field(default_factory=list)


@dataclass
class DebugInfo:
    r"""Debug information generated in running.

    Attributes:
        keys (set): Set of keys are pressed on keyboard.
    """

    keys: set = field(default_factory=set)


class BaseAgent(abc.ABC):  # pylint: disable=too-many-instance-attributes
    r"""Base class for agent.

    Get mujoco-specific info about agent and control agent in environments.

    Methods:

    - :meth:`_load_model`: Load agent model from xml file.
    - :meth:`_init_body_info`: Initialize body information.
    - :meth:`_build_action_space`: Build action space for agent.
    - :meth:`_init_jnt_sensors`: Initialize information of joint sensors in current agent.
    - :meth:`set_engine`: Set physical engine instance.
    - :meth:`apply_action`: Agent in physical simulator take specific action.
    - :meth:`build_sensor_observation_space`: Build agent specific observation space according to sensors.
    - :meth:`obs_sensor`: Get agent specific observations according to sensors.
    - :meth:`get_sensor`: Get specific sensor observations in agent.
    - :meth:`dist_xy`: Get distance between agent and target in XY plane.
    - :meth:`world_xy`: Get agent XY coordinate in world frame.
    - :meth:`keyboard_control_callback`: Keyboard control callback designed for debug mode for keyboard controlling.
    - :meth:`debug`: Implement specific action debug mode which maps keyboard input into action of agent.
    - :meth:`is_alive`: Check if agent is alive.
    - :meth:`reset`: Reset agent to specific initial internal state, eg.joints angles.

    Attributes:

    - :attr:`base` (str): Path to agent XML.
    - :attr:`random_generator` (RandomGenerator): Random generator.
    - :attr:`placements` (list): Agent placements list (defaults to full extents).
    - :attr:`locations` (list): Explicitly place agent XY coordinate.
    - :attr:`keepout` (float): Needs to be set to match the agent XML used.
    - :attr:`rot` (float): Override agent starting angle.
    - :attr:`engine` (:class:`Engine`): Physical engine instance.
    - :attr:`sensor_conf` (:class:`SensorConf`): Sensor observations configuration.
    - :attr:`sensor_info` (:class:`SensorInfo`): Sensor information.
    - :attr:`body_info` (:class:`BodyInfo`): Body information.
    - :attr:`debug_info` (:class:`DebugInfo`): Debug information.
    - :attr:`z_height` (float): Initial height of agent in environments.
    - :attr:`action_space` (:class:`gymnasium.spaces.Box`): Action space.
    - :attr:`com` (np.ndarray): The Cartesian coordinate of agent center of mass.
    - :attr:`mat` (np.ndarray): The Cartesian rotation matrix of agent.
    - :attr:`vel` (np.ndarray): The Cartesian velocity of agent.
    - :attr:`pos` (np.ndarray): The Cartesian position of agent.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        random_generator: RandomGenerator,
        placements: list | None = None,
        locations: list | None = None,
        keepout: float = 0.4,
        rot: float | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            name (str): Name of agent.
            random_generator (RandomGenerator): Random generator.
            placements (list): Agent placements list (defaults to full extents).
            locations (list): Explicitly place agent XY coordinate.
            keepout (float): Needs to be set to match the agent XML used.
            rot (float): Override agent starting angle.
        """
        self.base: str = f'assets/xmls/multi_{name.lower()}.xml'
        self.random_generator: RandomGenerator = random_generator
        self.placements: list = placements
        self.locations: list = [] if locations is None else locations
        self.keepout: float = keepout
        self.rot: float = rot
        self.possible_agents: list = ['agent_0', 'agent_1']
        self.nums: int = 2

        self.engine: Engine = None
        self._load_model()
        self.sensor_conf = SensorConf()
        self.sensor_info = SensorInfo()
        self.body_info = [BodyInfo(), BodyInfo()]
        self._init_body_info()
        self.debug_info = DebugInfo()

        # Needed to figure out z-height of free joint of offset body
        self.z_height: float = self.engine.data.body('agent').xpos[2]

        self.action_space: dict[gymnasium.spaces.Box] = self._build_action_space()
        self._init_jnt_sensors()

    def _load_model(self) -> None:
        """Load the agent model from the xml file.

        Note:
            The physical engine instance which is created here is just used to figure out the dynamics
            of agent and save some useful information, when the environment is actually created, the
            physical engine instance will be replaced by the new instance which is created in
            :class:`safety_gymnasium.World` via :meth:`set_engine`.
        """
        base_path = os.path.join(BASE_DIR, self.base)
        model = mujoco.MjModel.from_xml_path(base_path)  # pylint: disable=no-member
        data = mujoco.MjData(model)  # pylint: disable=no-member
        mujoco.mj_forward(model, data)  # pylint: disable=no-member
        self.set_engine(Engine(model, data))

    def _init_body_info(self) -> None:
        """Initialize body information.

        Access directly from mujoco instance created on agent xml model.
        """
        for i in range(2):
            self.body_info[i].nq = int(self.engine.model.nq / 2)
            self.body_info[i].nv = int(self.engine.model.nv / 2)
            self.body_info[i].nu = int(self.engine.model.nu / 2)
            self.body_info[i].nbody = int(self.engine.model.nbody / 2)
            self.body_info[i].geom_names = [
                self.engine.model.geom(i).name
                for i in range(self.engine.model.ngeom)
                if self.engine.model.geom(i).name != 'floor'
            ][i * 2 : (i + 1) * 2]

    def _build_action_space(self) -> gymnasium.spaces.Box:
        """Build the action space for this agent.

        Access directly from mujoco instance created on agent xml model.
        """
        bounds = self.engine.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        divide_index = int(len(low) / 2)
        return {
            'agent_0': spaces.Box(
                low=low[:divide_index],
                high=high[:divide_index],
                dtype=np.float64,
            ),
            'agent_1': spaces.Box(
                low=low[divide_index:],
                high=high[divide_index:],
                dtype=np.float64,
            ),
        }

    def _init_jnt_sensors(self) -> None:  # pylint: disable=too-many-branches
        """Initialize joint sensors.

        Access directly from mujoco instance created on agent xml model and save different
        joint names into different lists.
        """
        for i in range(self.engine.model.nsensor):
            name = self.engine.model.sensor(i).name
            sensor_id = self.engine.model.sensor(
                name,
            ).id  # pylint: disable=redefined-builtin, invalid-name
            self.sensor_info.sensor_dim[name] = self.engine.model.sensor(sensor_id).dim[0]
            sensor_type = self.engine.model.sensor(sensor_id).type
            if (
                # pylint: disable-next=no-member
                self.engine.model.sensor(sensor_id).objtype
                == mujoco.mjtObj.mjOBJ_JOINT  # pylint: disable=no-member
            ):  # pylint: disable=no-member
                joint_id = self.engine.model.sensor(sensor_id).objid
                joint_type = self.engine.model.jnt(joint_id).type
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE:  # pylint: disable=no-member
                    if sensor_type == mujoco.mjtSensor.mjSENS_JOINTPOS:  # pylint: disable=no-member
                        self.sensor_info.hinge_pos_names.append(name)
                    elif (
                        sensor_type == mujoco.mjtSensor.mjSENS_JOINTVEL
                    ):  # pylint: disable=no-member
                        self.sensor_info.hinge_vel_names.append(name)
                    else:
                        t = self.engine.model.sensor(i).type  # pylint: disable=invalid-name
                        raise ValueError(f'Unrecognized sensor type {t} for joint')
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:  # pylint: disable=no-member
                    if sensor_type == mujoco.mjtSensor.mjSENS_BALLQUAT:  # pylint: disable=no-member
                        self.sensor_info.ballquat_names.append(name)
                    elif (
                        sensor_type == mujoco.mjtSensor.mjSENS_BALLANGVEL
                    ):  # pylint: disable=no-member
                        self.sensor_info.ballangvel_names.append(name)
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:  # pylint: disable=no-member
                    # Adding slide joints is trivially easy in code,
                    # but this removes one of the good properties about our observations.
                    # (That we are invariant to relative whole-world transforms)
                    # If slide joints are added we should ensure this stays true!
                    raise ValueError('Slide joints in agents not currently supported')
            elif (
                # pylint: disable-next=no-member
                self.engine.model.sensor(sensor_id).objtype
                == mujoco.mjtObj.mjOBJ_SITE  # pylint: disable=no-member
            ):
                if name == 'agent_pos':
                    self.sensor_info.freejoint_pos_name = name
                elif name == 'agent_qvel':
                    self.sensor_info.freejoint_qvel_name = name

    def set_engine(self, engine: Engine) -> None:
        """Set the engine instance.

        Args:
            engine (Engine): The engine instance.

        Note:
            This method will be called twice in one single environment.
          1. When the agent is initialized, used to get and save useful information.
          2. When the environment is created, used to update the engine instance.
        """
        self.engine = engine

    def apply_action(self, action: np.ndarray, noise: np.ndarray | None = None) -> None:
        """Apply an action to the agent.

        Just fill up the control array in the engine data.

        Args:
            action (np.ndarray): The action to apply.
            noise (np.ndarray): The noise to add to the action.
        """
        action = np.array(action, copy=False)  # Cast to ndarray

        # Set action
        action_range = self.engine.model.actuator_ctrlrange

        self.engine.data.ctrl[:] = np.clip(action, action_range[:, 0], action_range[:, 1])

        if noise:
            self.engine.data.ctrl[:] += noise

    def build_sensor_observation_space(self) -> gymnasium.spaces.Dict:
        """Build observation space for all sensor types.

        Returns:
            gymnasium.spaces.Dict: The observation space generated by sensors bound with agent.
        """
        obs_space_dict = {}

        for sensor in self.sensor_conf.sensors:  # Explicitly listed sensors
            dim = self.sensor_info.sensor_dim[sensor]
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float64)
        # Velocities don't have wraparound effects that rotational positions do
        # Wraparounds are not kind to neural networks
        # Whereas the angle 2*pi is very close to 0, this isn't true in the network
        # In theory the network could learn this, but in practice we simplify it
        # when the sensors_angle_components switch is enabled.
        for sensor in self.sensor_info.hinge_vel_names:
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float64)
        for sensor in self.sensor_info.ballangvel_names:
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64)
        if self.sensor_info.freejoint_pos_name:
            sensor = self.sensor_info.freejoint_pos_name
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float64)
            obs_space_dict[sensor + '1'] = gymnasium.spaces.Box(
                -np.inf,
                np.inf,
                (1,),
                dtype=np.float64,
            )
        if self.sensor_info.freejoint_qvel_name:
            sensor = self.sensor_info.freejoint_qvel_name
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64)
            obs_space_dict[sensor + '1'] = gymnasium.spaces.Box(
                -np.inf,
                np.inf,
                (3,),
                dtype=np.float64,
            )
        # Angular positions have wraparound effects, so output something more friendly
        if self.sensor_conf.sensors_angle_components:
            # Single joints are turned into sin(x), cos(x) pairs
            # These should be easier to learn for neural networks,
            # Since for angles, small perturbations in angle give small differences in sin/cos
            for sensor in self.sensor_info.hinge_pos_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    (2,),
                    dtype=np.float64,
                )
            # Quaternions are turned into 3x3 rotation matrices
            # Quaternions have a wraparound issue in how they are normalized,
            # where the convention is to change the sign so the first element to be positive.
            # If the first element is close to 0, this can mean small differences in rotation
            # lead to large differences in value as the latter elements change sign.
            # This also means that the first element of the quaternion is not expectation zero.
            # The SO(3) rotation representation would be a good replacement here,
            # since it smoothly varies between values in all directions (the property we want),
            # but right now we have very little code to support SO(3) rotations.
            # Instead we use a 3x3 rotation matrix, which if normalized, smoothly varies as well.
            for sensor in self.sensor_info.ballquat_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    (3, 3),
                    dtype=np.float64,
                )
        else:
            # Otherwise include the sensor without any processing
            for sensor in self.sensor_info.hinge_pos_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    (1,),
                    dtype=np.float64,
                )
            for sensor in self.sensor_info.ballquat_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    (4,),
                    dtype=np.float64,
                )

        return obs_space_dict

    def obs_sensor(self) -> dict[str, np.ndarray]:
        """Get observations of all sensor types.

        Returns:
            Dict[str, np.ndarray]: The observations generated by sensors bound with agent.
        """
        obs = {}

        # Sensors which can be read directly, without processing
        for sensor in self.sensor_conf.sensors:  # Explicitly listed sensors
            obs[sensor] = self.get_sensor(sensor)
        for sensor in self.sensor_info.hinge_vel_names:
            obs[sensor] = self.get_sensor(sensor)
        for sensor in self.sensor_info.ballangvel_names:
            obs[sensor] = self.get_sensor(sensor)
        if self.sensor_info.freejoint_pos_name:
            sensor = self.sensor_info.freejoint_pos_name
            obs[sensor] = self.get_sensor(sensor)[2:]
            obs[sensor + '1'] = self.get_sensor(sensor + '1')[2:]
        if self.sensor_info.freejoint_qvel_name:
            sensor = self.sensor_info.freejoint_qvel_name
            obs[sensor] = self.get_sensor(sensor)
            obs[sensor + '1'] = self.get_sensor(sensor + '1')
        # Process angular position sensors
        if self.sensor_conf.sensors_angle_components:
            for sensor in self.sensor_info.hinge_pos_names:
                theta = float(self.get_sensor(sensor))  # Ensure not 1D, 1-element array
                obs[sensor] = np.array([np.sin(theta), np.cos(theta)])
            for sensor in self.sensor_info.ballquat_names:
                quat = self.get_sensor(sensor)
                obs[sensor] = quat2mat(quat)
        else:  # Otherwise read sensors directly
            for sensor in self.sensor_info.hinge_pos_names:
                obs[sensor] = self.get_sensor(sensor)
            for sensor in self.sensor_info.ballquat_names:
                obs[sensor] = self.get_sensor(sensor)

        return obs

    def get_sensor(self, name: str) -> np.ndarray:
        """Get the value of one sensor.

        Args:
            name (str): The name of the sensor to checkout.

        Returns:
            np.ndarray: The observation value of the sensor.
        """
        id = self.engine.model.sensor(name).id  # pylint: disable=redefined-builtin, invalid-name
        adr = self.engine.model.sensor_adr[id]
        dim = self.engine.model.sensor_dim[id]
        return self.engine.data.sensordata[adr : adr + dim].copy()

    def dist_xy(self, index, pos: np.ndarray) -> float:
        """Return the distance from the agent to an XY position.

        Args:
            pos (np.ndarray): The position to measure the distance to.

        Returns:
            float: The distance from the agent to the position.
        """
        pos = np.asarray(pos)
        if pos.shape == (3,):
            pos = pos[:2]
        if index == 0:
            agent_pos = self.pos_0
        elif index == 1:
            agent_pos = self.pos_1
        return np.sqrt(np.sum(np.square(pos - agent_pos[:2])))

    def world_xy(self, pos: np.ndarray) -> np.ndarray:
        """Return the world XY vector to a position from the agent.

        Args:
            pos (np.ndarray): The position to measure the vector to.

        Returns:
            np.ndarray: The world XY vector to the position.
        """
        assert pos.shape == (2,)
        return pos - self.agent.agent_pos()[:2]  # pylint: disable=no-member

    def keyboard_control_callback(self, key: int, action: int) -> None:
        """Callback for keyboard control.

        Collect keys which are pressed.

        Args:
            key (int): The key code inputted by user.
            action (int): The action of the key in glfw.
        """
        if action == glfw.PRESS:
            self.debug_info.keys.add(key)
        elif action == glfw.RELEASE:
            self.debug_info.keys.remove(key)

    def debug(self) -> None:
        """Debug mode.

        Apply action which is inputted from keyboard.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_alive(self) -> bool:
        """Returns True if the agent is healthy.

        Returns:
            bool: True if the agent is healthy,
                False if the agent is unhealthy.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Called when the environment is reset."""

    @property
    def com(self) -> np.ndarray:
        """Get the position of the agent center of mass in the simulator world reference frame.

        Returns:
            np.ndarray: The Cartesian position of the agent center of mass.
        """
        return self.engine.data.body('agent').subtree_com.copy()

    @property
    def mat_0(self) -> np.ndarray:
        """Get the rotation matrix of the agent in the simulator world reference frame.

        Returns:
            np.ndarray: The Cartesian rotation matrix of the agent.
        """
        return self.engine.data.body('agent').xmat.copy().reshape(3, -1)

    @property
    def mat_1(self) -> np.ndarray:
        """Get the rotation matrix of the agent in the simulator world reference frame.

        Returns:
            np.ndarray: The Cartesian rotation matrix of the agent.
        """
        return self.engine.data.body('agent1').xmat.copy().reshape(3, -1)

    @property
    def vel(self) -> np.ndarray:
        """Get the velocity of the agent in the simulator world reference frame.

        Returns:
            np.ndarray: The velocity of the agent.
        """
        return get_body_xvelp(self.engine.model, self.engine.data, 'agent').copy()

    @property
    def pos_0(self) -> np.ndarray:
        """Get the position of the agent in the simulator world reference frame.

        Returns:
            np.ndarray: The Cartesian position of the agent.
        """
        return self.engine.data.body('agent').xpos.copy()

    @property
    def pos_1(self) -> np.ndarray:
        """Get the position of the agent in the simulator world reference frame.

        Returns:
            np.ndarray: The Cartesian position of the agent.
        """
        return self.engine.data.body('agent1').xpos.copy()
