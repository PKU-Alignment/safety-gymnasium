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


"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU) -> None:
        super().__init__()
        self._layer_N = layer_N

        # active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        active_func = [nn.ELU(), nn.ELU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        # self.fc1 = nn.Sequential(
        #     init_(nn.Linear(input_dim, hidden_size)), active_func)
        # self.fc_h = nn.Sequential(init_(
        #     nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # self.fc2 = get_clones(self.fc_h, self._layer_N)
        self.fc2 = nn.ModuleList(
            [
                nn.Sequential(
                    init_(nn.Linear(hidden_size, hidden_size)),
                    active_func,
                    nn.LayerNorm(hidden_size),
                )
                for i in range(self._layer_N)
            ],
        )
        # self.fc2 = nn.ModuleList([nn.Sequential(init_(
        #     nn.Linear(hidden_size, hidden_size)), active_func) for i in range(self._layer_N)])

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, config, obs_shape, cat_self=True, attn_internal=False) -> None:
        super().__init__()

        self._use_feature_normalization = config['use_feature_normalization']
        self._use_orthogonal = config['use_orthogonal']
        self._use_ReLU = config['use_ReLU']
        self._stacked_frames = config['stacked_frames']
        self._layer_N = config['layer_N']
        self.hidden_size = config['hidden_size']

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim,
            self.hidden_size,
            self._layer_N,
            self._use_orthogonal,
            self._use_ReLU,
        )
        # self.mlp_middle_layer = MLPLayer(self.hidden_size, self.hidden_size,
        #                       self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        return self.mlp(x)
        # x = self.mlp_middle_layer(x)
