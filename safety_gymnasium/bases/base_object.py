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
"""Base class for obstacles."""

import abc
from dataclasses import dataclass

import numpy as np

from safety_gymnasium.bases.base_agent import BaseAgent
from safety_gymnasium.utils.random_generator import RandomGenerator
from safety_gymnasium.world import Engine


@dataclass
class BaseObject(abc.ABC):
    r"""Base class for obstacles.

    Methods:

    - :meth:`cal_cost`: Calculate the cost of the object, only when the object can be constrained, it
      is needed to be implemented.
    - :meth:`set_agent`: Set the agent instance, only called once for each object in one environment.
    - :meth:`set_engine`: Set the engine instance, only called once in :class:`safety_gymnasium.World`.
    - :meth:`set_random_generator`: Set the random generator instance, only called once in one environment.
    - :meth:`process_config`: Process the config, used to fill the configuration dictionary which used to
      generate mujoco instance xml string of environments.
    - :meth:`_specific_agent_config`: Modify properties according to specific agent.
    - :meth:`get_config`: Define how to generate config of different objects, it will be called in process_config.

    Attributes:

    - :attr:`type` (str): Type of the obstacle, used as key in :meth:`process_config` to fill configuration
      dictionary.
    - :attr:`name` (str): Name of the obstacle, used as key in :meth:`process_config` to fill configuration
      dictionary.
    - :attr:`engine` (:class:`safety_gymnasium.world.Engine`): Physical engine instance.
    - :attr:`random_generator` (:class:`safety_gymnasium.utils.random_generator.RandomGenerator`):
      Random generator instance.
    - :attr:`agent` (:class:`safety_gymnasium.bases.base_agent.BaseAgent`): Agent instance.
    - :attr:`pos` (np.ndarray): Get the position of the object.
    """

    type: str = None
    name: str = None
    engine: Engine = None
    random_generator: RandomGenerator = None
    agent: BaseAgent = None

    def cal_cost(self) -> dict:
        """Calculate the cost of the obstacle.

        Returns:
            dict: Cost of the object in current environments at this timestep.
        """
        return {}

    def set_agent(self, agent: BaseAgent) -> None:
        """Set the agent instance.

        Note:
            This method will be called only once in one environment, that is when the object
            is instantiated.

        Args:
            agent (BaseAgent): Agent instance in current environment.
        """
        self.agent = agent
        self._specific_agent_config()

    def set_engine(self, engine: Engine) -> None:
        """Set the engine instance.

        Note:
            This method will be called only once in one environment, that is when the whole
            environment is instantiated in :meth:`safety_gymnasium.World.bind_engine`.

        Args:
            engine (Engine): Physical engine instance.
        """
        self.engine = engine

    def set_random_generator(self, random_generator: RandomGenerator) -> None:
        """Set the random generator instance.

        Args:
            random_generator (RandomGenerator): Random generator instance.
        """
        self.random_generator = random_generator

    def process_config(self, config: dict, layout: dict, rots: float) -> None:
        """Process the config.

        Note:
            This method is called in :meth:`safety_gymnasium.bases.base_task._build_world_config` to
            fill the configuration dictionary which used to generate mujoco instance xml string of
            environments in :meth:`safety_gymnasium.World.build`.
        """
        if hasattr(self, 'num'):
            assert (
                len(rots) == self.num
            ), 'The number of rotations should be equal to the number of obstacles.'
            for i in range(self.num):
                name = f'{self.name[:-1]}{i}'
                config[self.type][name] = self.get_config(xy_pos=layout[name], rot=rots[i])
                config[self.type][name].update({'name': name})
                config[self.type][name]['geoms'][0].update({'name': name})
        else:
            assert len(rots) == 1, 'The number of rotations should be 1.'
            config[self.type][self.name] = self.get_config(xy_pos=layout[self.name], rot=rots[0])

    def _specific_agent_config(self) -> None:  # noqa: B027
        """Modify properties according to specific agent.

        Note:
            This method will be called only once in one environment, that is when :meth:`set_agent`
            is called.
        """

    @property
    @abc.abstractmethod
    def pos(self) -> np.ndarray:
        """Get the position of the obstacle.

        Returns:
            np.ndarray: Position of the obstacle.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_config(self, xy_pos: np.ndarray, rot: float):
        """Get the config of the obstacle.

        Returns:
            dict: Configuration of this type of object in current environment.
        """
        raise NotImplementedError


@dataclass
class Geom(BaseObject):
    r"""Base class for obstacles that are geoms.

    Attributes:
        type (str): Type of the object, used as key in :meth:`process_config` to fill configuration
            dictionary.
    """

    type: str = 'geoms'


@dataclass
class FreeGeom(BaseObject):
    r"""Base class for obstacles that are objects.

    Attributes:
        type (str): Type of the object, used as key in :meth:`process_config` to fill configuration
            dictionary.
    """

    type: str = 'free_geoms'


@dataclass
class Mocap(BaseObject):
    r"""Base class for obstacles that are mocaps.

    Attributes:
        type (str): Type of the object, used as key in :meth:`process_config` to fill configuration
            dictionary.
    """

    type: str = 'mocaps'

    def process_config(self, config: dict, layout: dict, rots: float) -> None:
        """Process the config.

        Note:
            This method is called in :meth:`safety_gymnasium.bases.base_task._build_world_config` to
            fill the configuration dictionary which used to generate mujoco instance xml string of
            environments in :meth:`safety_gymnasium.World.build`.
            As Mocap type object, it will generate two objects, one is the mocap object, the other
            is the object that is attached to the mocap object, this is due to the mocap's mechanism
            of mujoco.
        """
        if hasattr(self, 'num'):
            assert (
                len(rots) == self.num
            ), 'The number of rotations should be equal to the number of obstacles.'
            for i in range(self.num):
                mocap_name = f'{self.name[:-1]}{i}mocap'
                obj_name = f'{self.name[:-1]}{i}obj'
                layout_name = f'{self.name[:-1]}{i}'
                configs = self.get_config(xy_pos=layout[layout_name], rot=rots[i])
                config['free_geoms'][obj_name] = configs['obj']
                config['free_geoms'][obj_name].update({'name': obj_name})
                config['free_geoms'][obj_name]['geoms'][0].update({'name': obj_name})
                config['mocaps'][mocap_name] = configs['mocap']
                config['mocaps'][mocap_name].update({'name': mocap_name})
                config['mocaps'][mocap_name]['geoms'][0].update({'name': mocap_name})
        else:
            assert len(rots) == 1, 'The number of rotations should be 1.'
            mocap_name = f'{self.name[:-1]}mocap'
            obj_name = f'{self.name[:-1]}obj'
            layout_name = self.name[:-1]
            configs = self.get_config(xy_pos=layout[layout_name], rot=rots[0])
            config['free_geoms'][obj_name] = configs['obj']
            config['free_geoms'][obj_name].update({'name': obj_name})
            config['free_geoms'][obj_name]['geoms'][0].update({'name': obj_name})
            config['mocaps'][mocap_name] = configs['mocap']
            config['mocaps'][mocap_name].update({'name': mocap_name})
            config['mocaps'][mocap_name]['geoms'][0].update({'name': mocap_name})

    def set_mocap_pos(self, name: str, value: np.ndarray) -> None:
        """Set the position of a mocap object.

        Args:
            name (str): Name of the mocap object.
            value (np.ndarray): Target position of the mocap object.
        """
        body_id = self.engine.model.body(name).id
        mocap_id = self.engine.model.body_mocapid[body_id]
        self.engine.data.mocap_pos[mocap_id] = value

    @abc.abstractmethod
    def move(self) -> None:
        """Set mocap object positions before a physics step is executed.

        Note:
            This method is called in :meth:`safety_gymnasium.bases.base_task.simulation_forward` before a physics
            step is executed.
        """
