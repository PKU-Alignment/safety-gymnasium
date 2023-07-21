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

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class Actor(nn.Module):
    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        model_cfg,
        asymmetric=False,
    ) -> None:
        super().__init__()

        self.asymmetric = asymmetric

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            activation = get_activation('selu')
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(*obs_shape, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        # self.logstd_layer = nn.Parameter(torch.ones(1, act_dim) * torch.tensor(log_std))
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))
        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        self.init_weights(self.actor, actor_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):
        actions_mean = self.actor(observations)
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)
        logstd = self.log_std.repeat(actions_mean.shape[0], 1)
        action_std = torch.exp(logstd)
        return (
            actions.detach(),
            actions_log_prob.detach(),
            actions_mean.detach(),
            action_std.detach(),
        )

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, observations, states, actions):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        actions_log_prob = distribution.log_prob(actions)
        logstd = self.log_std.repeat(actions_mean.shape[0], 1)
        action_std = torch.exp(logstd)

        return actions_log_prob, entropy, actions_mean, action_std


class Critic(nn.Module):
    def __init__(self, obs_shape, states_shape, model_cfg, asymmetric=False) -> None:
        super().__init__()

        self.asymmetric = asymmetric

        if model_cfg is None:
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation('selu')
        else:
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Value function
        critic_layers = []
        if self.asymmetric:
            critic_layers.append(nn.Linear(*states_shape, critic_hidden_dim[0]))
        else:
            critic_layers.append(nn.Linear(*obs_shape, critic_hidden_dim[0]))

        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        # print(self.critic)

        # Initialize the weights like in stable baselines
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)

        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):
        value = self.critic(states) if self.asymmetric else self.critic(observations)

        return value.detach()

    def evaluate(self, observations, states):
        return self.critic(states) if self.asymmetric else self.critic(observations)


def get_activation(act_name):
    if act_name == 'elu':
        return nn.ELU()
    elif act_name == 'selu':
        return nn.SELU()
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'crelu':
        return nn.ReLU()
    elif act_name == 'lrelu':
        return nn.LeakyReLU()
    elif act_name == 'tanh':
        return nn.Tanh()
    elif act_name == 'sigmoid':
        return nn.Sigmoid()
    else:
        print('invalid activation function!')
        return None
