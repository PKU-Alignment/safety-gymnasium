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
"""Utils for common usage."""

from __future__ import annotations

import re

import numpy as np
import xmltodict


def quat2zalign(quat):
    """From quaternion, extract z_{ground} dot z_{body}"""
    # z_{body} from quaternion [a,b,c,d] in ground frame is:
    # [ 2bd + 2ac,
    #   2cd - 2ab,
    #   a**2 - b**2 - c**2 + d**2
    # ]
    # so inner product with z_{ground} = [0,0,1] is
    # z_{body} dot z_{ground} = a**2 - b**2 - c**2 + d**2
    a, b, c, d = quat  # pylint: disable=invalid-name
    return a**2 - b**2 - c**2 + d**2


def convert(value):
    """Convert a value into a string for mujoco XML."""
    if isinstance(value, (int, float, str)):
        return str(value)
    # Numpy arrays and lists
    return ' '.join(str(i) for i in np.asarray(value))


def rot2quat(theta):
    """Get a quaternion rotated only about the Z axis."""
    return np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)], dtype='float64')


def camel_to_snake(name):
    """Convert a camel case name to snake case."""
    string = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()


def build_xml_from_dict(data):
    """Build an XML string from a dictionary then parse it with xmltodict."""
    body_attrs = ' '.join(
        [f'{k}="{convert(v)}"' for k, v in data.items() if k not in ['geom', 'geoms', 'freejoint']],
    )
    geoms = data.get('geoms', [data.get('geom')]) if data.get('geom') or data.get('geoms') else []
    freejoint = data.get('freejoint')

    geoms_xml = ''
    for geom in geoms:
        geom_attrs = ' '.join([f'{k}="{convert(v)}"' for k, v in geom.items()])
        geoms_xml += f'<geom {geom_attrs}/>'

    freejoint_xml = ''
    if freejoint:
        freejoint_xml = f'<freejoint name="{freejoint}"/>'

    xml_string = f'<body {body_attrs}>{freejoint_xml}{geoms_xml}</body>'
    return xmltodict.parse(xml_string)


class ResamplingError(AssertionError):
    """Raised when we fail to sample a valid distribution of objects or goals."""


class MujocoException(Exception):
    """Raise when mujoco raise an exception during simulation."""
