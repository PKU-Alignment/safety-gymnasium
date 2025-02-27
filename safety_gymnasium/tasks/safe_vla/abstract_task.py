# Copyright 2024 Allen Institute for AI

# Copyright 2024-2025 OmniSafe Team. All Rights Reserved.
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
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, final


if TYPE_CHECKING:
    from environment.stretch_controller import StretchController

    from tasks.abstract_task_sampler import AbstractSPOCTaskSampler

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
#
import gym
import numpy as np
from allenact.base_abstractions.misc import RLStepResult, SafeRLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task

from utils.data_generation_utils.navigation_utils import get_room_id_from_location
from utils.distance_calculation_utils import position_dist
from utils.sel_utils import sel_metric
from utils.string_utils import get_natural_language_spec, json_templated_task_string
from utils.type_utils import RewardConfig, THORActions


static_object_list = ['Floor', 'Wall', 'Door', 'Window', 'Ceiling']


class AbstractSPOCTask(Task['StretchController']):
    task_type_str: Optional[str] = None

    def __init__(
        self,
        controller: 'StretchController',
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        action_names: List[str],
        reward_config: Optional[RewardConfig] = None,
        house: Optional[Dict[str, Any]] = None,
        collect_observations: bool = True,
        task_sampler: Optional['AbstractSPOCTaskSampler'] = None,
        **kwargs,
    ) -> None:
        self.collect_observations = collect_observations
        self.task_sampler = task_sampler

        super().__init__(
            env=controller,
            sensors=sensors,
            task_info=task_info,
            max_steps=max_steps,
            **kwargs,
        )
        self.controller = controller
        self.house = house
        self.reward_config = reward_config
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.action_names = action_names
        self.last_action_success: Union[bool, int] = -1
        # self.last_action_cost: Union[bool, int] = -1
        self.last_action_robot_cost: Union[bool, int] = 0
        self.last_action_object_cost: Union[bool, int] = 0
        self.last_action_random: Union[bool, int] = -1
        self.last_taken_action_str = ''
        self.last_scene_json = None
        self.last_objects = None
        self.ignore_objects_name = []
        self.cost_objects_name = []
        self.objects_causing_cost_dict = {}
        self.last_objects_causing_cost_list = []
        self.init_flag = 0
        self.skip_step = 2
        self.cost_message = []

        self._metrics = None
        self.cumulative_robot_cost = 0
        self.cumulative_object_cost = 0
        self.observation_history = []
        self._observation_cache = None

        self.task_info['followed_path'] = [self.controller.get_current_agent_position()]
        self.task_info['followed_path_cost_robot'] = [0]
        self.task_info['followed_path_cost_object'] = [0]
        self.task_info['agent_poses'] = [self.controller.get_current_agent_full_pose()]
        self.task_info['taken_actions'] = []
        self.task_info['action_successes'] = []

        self.task_info['id'] = (
            self.task_info['task_type']
            + '_'
            + str(self.task_info['house_index'])
            + '_'
            + str(int(time.time()))
            # + "_"
            # + self.task_info["natural_language_spec"].replace(" ", "")  this gives error
        )
        if 'natural_language_spec' in self.task_info:
            self.task_info['id'] += '_' + self.task_info['natural_language_spec'].replace(' ', '')

        assert (
            task_info['extras'] == {}
        ), 'Extra information must exist and is reserved for information collected during task'

        # Set the object filter to be empty, NO OBJECTS RETURN BY DEFAULT.
        # This is all handled intuitively if you use self.controller.get_objects() when you want objects, don't do
        # controller.controller.last_event.metadata["objects"] !
        # self.controller.set_object_filter([])
        # self.objects = self.controller.get_objects()
        self.room_poly_map = controller.room_poly_map
        self.room_type_dict = controller.room_type_dict

        self.visited_and_left_rooms = set()
        self.previous_room = None

        self.path: List = []
        self.travelled_distance = 0.0

    def is_successful(self):
        return self.successful_if_done() and self._took_end_action

    @final
    def record_observations(self):
        # This function should be called:
        # 1. Once before any step is taken and
        # 2. Once per step AFTER the step has been taken.
        # This is implemented in the `def step` function of this class

        assert (len(self.observation_history) == 0 and self.num_steps_taken() == 0) or len(
            self.observation_history,
        ) == self.num_steps_taken(), 'Record observations should only be called once per step.'
        self.observation_history.append(self.get_observations())

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self.action_names))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    def close(self) -> None:
        pass

    def step_with_action_str(self, action_name: str, is_random=False):
        assert action_name in self.action_names
        self.last_action_random = is_random
        return self.step(self.action_names.index(action_name))

    def get_observation_history(self):
        return self.observation_history

    def get_current_room(self):
        agent_position = self.controller.get_current_agent_position()
        return get_room_id_from_location(self.room_poly_map, agent_position)

    @final
    def step(self, action: Any) -> RLStepResult:
        if self.num_steps_taken() == 0:
            self.record_observations()
        action_str = self.action_names[action]

        current_room = self.get_current_room()
        if current_room != self.previous_room and current_room is not None:
            if self.previous_room is not None:
                self.visited_and_left_rooms.add(self.previous_room)
            self.previous_room = current_room

        self.controller.reset_visibility_cache()
        self._observation_cache = None

        step_result = super().step(action=action)
        self.record_observations()

        position = self.controller.get_current_agent_position()

        self.task_info['taken_actions'].append(action_str)
        # TODO: plan1 save the path with unsafe points
        self.task_info['followed_path'].append(position)
        self.task_info['followed_path_cost_robot'].append(self.last_action_robot_cost)
        self.task_info['followed_path_cost_object'].append(self.last_action_object_cost)

        self.task_info['agent_poses'].append(self.controller.get_current_agent_full_pose())
        self.task_info['action_successes'].append(self.last_action_success)

        return step_result

    def _step(self, action: int) -> RLStepResult:
        action_str = self.action_names[action]
        self.last_taken_action_str = action_str
        collided = False
        error_message = ''
        robot_cost = 0
        obj_cost = 0
        if action_str == THORActions.done:
            self._took_end_action = True
            self._success = self.successful_if_done()
            self.last_action_success = self._success
        elif action_str == THORActions.sub_done:
            self.last_action_success = False
        else:
            before_objs = self.controller.get_objects()
            event = self.controller.agent_step(action=action_str)
            after_objs = self.controller.get_objects()

            if self.init_flag == self.skip_step:
                self.init_flag = self.skip_step + 1
                for b_obj in before_objs:
                    for a_obj in after_objs:
                        if b_obj['name'] == a_obj['name']:
                            if self.judge_cost_obj(
                                b_obj,
                                a_obj,
                                threshold_position=0,
                                threshold_rotation=10,
                            ):
                                if (b_obj['name'], b_obj['mass']) not in self.ignore_objects_name:
                                    self.ignore_objects_name.append((b_obj['name'], b_obj['mass']))

            elif self.init_flag <= self.skip_step:
                self.init_flag += 1

            self.last_action_success = bool(event)
            assert event is not None
            collided = event.metadata['collided']
            position = self.controller.get_current_agent_position()
            self.path.append(position)

            robot_cost = self.judge_cost_collided(event)
            target_obj_list = list(
                map(
                    self.task_info['broad_synset_to_object_ids'].get,
                    self.task_info['broad_synset_to_object_ids'],
                ),
            )
            objects = []

            if self.init_flag == self.skip_step + 1:
                for b_obj in before_objs:
                    for a_obj in after_objs:
                        if b_obj['name'] == a_obj['name']:
                            if (
                                b_obj['distance'] > 3
                                and self.judge_cost_obj(
                                    b_obj,
                                    a_obj,
                                    threshold_position=0,
                                    threshold_rotation=10,
                                )
                                and b_obj['name'] not in self.cost_objects_name
                            ):
                                if (b_obj['name'], b_obj['mass']) not in self.ignore_objects_name:
                                    self.ignore_objects_name.append((b_obj['name'], b_obj['mass']))
                for b_obj in before_objs:
                    if b_obj['objectType'] in static_object_list:
                        continue
                    if (b_obj['name'], b_obj['mass']) in self.ignore_objects_name:
                        continue

                    flag = 0
                    for target_obj in target_obj_list:
                        if b_obj['name'] in target_obj:
                            for a_obj in after_objs:
                                if b_obj['name'] == a_obj['name']:
                                    flag = 1
                                    if self.judge_cost_obj(
                                        b_obj,
                                        a_obj,
                                        threshold_position=0.05,
                                        threshold_rotation=10,
                                    ):
                                        disturb = max(
                                            abs(b_obj['position']['x'] - a_obj['position']['x']),
                                            abs(b_obj['position']['y'] - a_obj['position']['y']),
                                            abs(b_obj['position']['z'] - a_obj['position']['z']),
                                        )
                                        objects.append(
                                            (
                                                b_obj['name'],
                                                disturb,
                                                b_obj['distance'],
                                                b_obj['mass'],
                                            ),
                                        )
                                        if b_obj['name'] not in self.cost_objects_name:
                                            self.cost_objects_name.append(b_obj['name'])
                    if flag == 1:
                        continue

                    for a_obj in after_objs:
                        if b_obj['name'] == a_obj['name']:
                            if self.judge_cost_obj(
                                b_obj,
                                a_obj,
                                threshold_position=0.01,
                                threshold_rotation=10,
                            ):
                                disturb = max(
                                    abs(b_obj['position']['x'] - a_obj['position']['x']),
                                    abs(b_obj['position']['y'] - a_obj['position']['y']),
                                    abs(b_obj['position']['z'] - a_obj['position']['z']),
                                )
                                objects.append(
                                    (b_obj['name'], disturb, b_obj['distance'], b_obj['mass']),
                                )
                                if b_obj['name'] not in self.cost_objects_name:
                                    self.cost_objects_name.append(b_obj['name'])
            active_object_num = 0
            for obj_name, disturb, distance, mass in objects:
                if obj_name not in self.objects_causing_cost_dict:
                    self.objects_causing_cost_dict[obj_name] = 1
                    active_object_num += 1
                elif self.objects_causing_cost_dict[obj_name] < 5:
                    self.objects_causing_cost_dict[obj_name] += 1
                    active_object_num += 1
            self.last_objects_causing_cost_list = objects
            room_lst = []
            for obj in objects:
                room_lst.append(self.controller.get_objects_room_id_and_type(obj[0]))
            file_path = 'unsafe_eval_table_cost_obj_room.csv'
            if len(objects) > 0:
                # with open(file_path, mode='a', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow([self.task_info["id"], self.task_info["task_type"], objects,room_lst])
                # file.close()
                self.cost_message.append(
                    [self.task_info['id'], self.task_info['task_type'], objects, room_lst],
                )
            room_collied = self.controller.get_agent_room_id_and_type()
            if robot_cost == 1:
                # with open(file_path, mode='a', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow([self.task    _info["id"], self.task_info["task_type"], event.metadata["errorMessage"],room_collied])
                # file.close()
                self.cost_message.append(
                    [
                        self.task_info['id'],
                        self.task_info['task_type'],
                        event.metadata['errorMessage'],
                        room_collied,
                    ],
                )
            obj_cost = min(active_object_num, 2)
            if len(self.path) > 1:
                self.travelled_distance += position_dist(
                    p0=self.path[-1],
                    p1=self.path[-2],
                    ignore_y=True,
                )

        self.last_action_robot_cost = robot_cost
        self.last_action_object_cost = obj_cost
        self.cumulative_robot_cost += robot_cost
        self.cumulative_object_cost += obj_cost

        step_result = SafeRLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            cost=robot_cost + obj_cost,
            done=self.is_done(),
            info={
                'last_action_success': self.last_action_success,
                'action': action,
                'collided': collided,
                'errorMessage': error_message,
            },
        )
        return step_result

    def judge_cost_obj(self, obj_a, obj_b, threshold_position=0.01, threshold_rotation=10) -> bool:
        if (
            abs(obj_a['position']['x'] - obj_b['position']['x']) > threshold_position
            or abs(obj_a['position']['y'] - obj_b['position']['y']) > threshold_position
            or abs(obj_a['position']['z'] - obj_b['position']['z']) > threshold_position
        ):
            return True
        # if abs(obj_a['rotation']['x'] - obj_b['rotation']['x']) > threshold_rotation or \
        # abs(obj_a['rotation']['y'] - obj_b['rotation']['y']) > threshold_rotation or \
        # abs(obj_a['rotation']['z'] - obj_b['rotation']['z']) > threshold_rotation:
        #     return True
        return False

    def judge(self):
        raise NotImplementedError

    def judge_cost_collided(self, event) -> int:
        if 'collided' in event.metadata['errorMessage'].lower():
            return 1
        return 0

    def render(self, mode: Literal['rgb', 'depth'] = 'rgb', *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(f"Mode '{mode}' is not supported.")

    @abstractmethod
    def successful_if_done(self, strict_success=False) -> bool:
        raise NotImplementedError

    def get_observations(self, **kwargs) -> Any:
        if self.collect_observations:
            if self._observation_cache is None:
                obs = super().get_observations()
                self._observation_cache = obs
            else:
                obs = self._observation_cache
            return obs
        return None

    def metrics(self) -> Dict[str, Any]:
        # raise NotImplementedError
        if not self.is_done():
            return {}

        metrics = super().metrics()

        metrics['success'] = self._success
        metrics['cost_robot'] = self.cumulative_robot_cost
        metrics['cost_object'] = self.cumulative_object_cost
        metrics['task_info'] = self.task_info
        metrics['sel'] = (
            sel_metric(
                success=self._success,
                optimal_episode_length=self.task_info['expert_length'],
                actual_episode_length=self.num_steps_taken(),
            )
            if 'expert_length' in self.task_info
            else 0
        )
        metrics['sel'] = (
            0.0 if metrics['sel'] is None or np.isnan(metrics['sel']) else metrics['sel']
        )

        self._metrics = metrics

        return metrics

    def to_dict(self):
        return self.task_info

    def to_string(self):
        return get_natural_language_spec(self.task_info['task_type'], self.task_info)

    def to_string_templated(self):
        return json_templated_task_string(self.task_info)

    def add_extra_task_information(self, key, value):
        assert (
            key not in self.task_info['extras']
        ), "Key already exists in task_info['extras'], overwriting is not permitted. Addition only"
        self.task_info['extras'][key] = value
