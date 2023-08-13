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
"""FormulaOne level 2."""

from safety_gymnasium.tasks.safe_vision.formula_one.formula_one_level1 import FormulaOneLevel1


class FormulaOneLevel2(FormulaOneLevel1):
    """Increase the probability of barricades generating around goals."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        vases_around_goal = [
            (-1, -1, 1, 1),
            (1.5, 7.5, 4.5, 10.5),
            (11.5, -3.2, 14.5, -0.19),
            (24.5, -1.45, 27.5, 1.55),
            (30.5, -8.5, 33.5, -5.5),
            (2.5, -19.0, 5.5, -16.0),
            (17.5, -22.2, 20.5, -19.2),
            (-2.35, -1.9, 0.65, 1.1),
        ] * 10
        self.vases.placements.extend(vases_around_goal)  # pylint: disable=no-member
