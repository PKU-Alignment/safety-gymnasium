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

import torch.nn as nn

from .util import init


"""CNN Modules and utils."""


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    def __init__(
        self,
        obs_shape,
        hidden_size,
        use_orthogonal,
        use_ReLU,
        kernel_size=3,
        stride=1,
    ) -> None:
        super().__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(
                nn.Conv2d(
                    in_channels=input_channel,
                    out_channels=hidden_size // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ),
            active_func,
            Flatten(),
            init_(
                nn.Linear(
                    hidden_size
                    // 2
                    * (input_width - kernel_size + stride)
                    * (input_height - kernel_size + stride),
                    hidden_size,
                ),
            ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
        )

    def forward(self, x):
        x = x / 255.0
        return self.cnn(x)


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape) -> None:
        super().__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        return self.cnn(x)
