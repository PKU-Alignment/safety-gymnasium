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
from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules
from typing import Dict, Type

from safety_gymnasium.tasks.safe_vla.abstract_task import AbstractSPOCTask
from safety_gymnasium.tasks.safe_vla.abstract_task_sampler import AbstractSPOCTaskSampler
from utils.type_utils import REGISTERED_TASK_PARAMS


REGISTERED_TASKS: Dict[str, Type[AbstractSPOCTask]] = {}


def register_task(cls):
    if cls.task_type_str not in REGISTERED_TASK_PARAMS:
        return cls

    REGISTERED_TASKS[cls.task_type_str] = cls
    return cls


# iterate through the modules in the current package
package_dir = str(Path(__file__).resolve().parent)
for _, module_name, _ in iter_modules([package_dir]):
    # import the module and iterate through its attributes
    module = import_module(f'{__name__}.{module_name}')
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isclass(attribute):
            # Add the class to this package's variables
            if issubclass(attribute, AbstractSPOCTask) and attribute != AbstractSPOCTask:
                globals()[attribute_name] = attribute
                register_task(attribute)
