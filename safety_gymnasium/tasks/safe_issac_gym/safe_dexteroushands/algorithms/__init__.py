# Copyright 2023 OmniSafe Team. All Rights Reserved.
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

from algorithms.cpo import CPO
from algorithms.focops import FOCOPS
from algorithms.p3o import P3O
from algorithms.pcpo import PCPO
from algorithms.ppol import PPOL
from algorithms.trpol import TRPOL


REGISTRY = {
    'cpo': CPO,
    'ppol': PPOL,
    'trpol': TRPOL,
    'focops': FOCOPS,
    'pcpo': PCPO,
    'p3o': P3O,
}
