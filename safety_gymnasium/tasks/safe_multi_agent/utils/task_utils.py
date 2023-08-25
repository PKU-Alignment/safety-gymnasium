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
"""Utils for task classes."""

import re

import mujoco
import numpy as np

import xml.etree.ElementTree as ET
from copy import deepcopy


def get_task_class_name(task_id):
    """Help to translate task_id into task_class_name."""
    class_name = ''.join(re.findall('[A-Z][^A-Z]*', task_id.split('-')[0])[2:])
    return class_name[:-1] + 'Level' + class_name[-1]


def quat2mat(quat):
    """Convert Quaternion to a 3x3 Rotation Matrix using mujoco."""
    # pylint: disable=invalid-name
    q = np.array(quat, dtype='float64')
    m = np.zeros(9, dtype='float64')
    mujoco.mju_quat2Mat(m, q)  # pylint: disable=no-member
    return m.reshape((3, 3))


def theta2vec(theta):
    """Convert an angle (in radians) to a unit vector in that angle around Z"""
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def get_body_jacp(model, data, name, jacp=None):
    """Get specific body's Jacobian via mujoco."""
    id = model.body(name).id  # pylint: disable=redefined-builtin, invalid-name
    if jacp is None:
        jacp = np.zeros(3 * model.nv).reshape(3, model.nv)
    jacp_view = jacp
    mujoco.mj_jacBody(model, data, jacp_view, None, id)  # pylint: disable=no-member
    return jacp


def get_body_xvelp(model, data, name):
    """Get specific body's Cartesian velocity."""
    jacp = get_body_jacp(model, data, name).reshape((3, model.nv))
    return np.dot(jacp, data.qvel)


def add_velocity_marker(viewer, pos, vel, cost, velocity_threshold):
    """Add a marker to the viewer to indicate the velocity of the agent."""
    pos = pos + np.array([0, 0, 0.6])
    safe_color = np.array([0.2, 0.8, 0.2, 0.5])
    unsafe_color = np.array([0.5, 0, 0, 0.5])

    if cost:
        color = unsafe_color
    else:
        vel_ratio = vel / velocity_threshold
        color = safe_color * (1 - vel_ratio)

    viewer.add_marker(
        pos=pos,
        size=0.2 * np.ones(3),
        type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
        rgba=color,
        label='',
    )


def clear_viewer(viewer):
    """Clear the viewer's all markers and overlays."""
    # pylint: disable=protected-access
    viewer._markers[:] = []
    viewer._overlays.clear()


def generate_agents(xml_content, num_agents):
    # Parse the XML content
    root = ET.fromstring(xml_content)
    
    # Find the agent body, actuator, and sensor
    agent_body = None
    for child in root.find('worldbody'):
        if child.get('name') == 'agent':
            agent_body = child
            break
    
    actuators = root.find('actuator')
    sensors = root.find('sensor')
    equalitys = root.find('equality')
    contacts = root.find('contact')
    
    # Lists to store the original actuators and sensors
    original_actuators = list(actuators) if actuators is not None else []
    original_sensors = list(sensors) if sensors is not None else []
    original_equality = list(equalitys) if equalitys is not None else []
    original_contact = list(contacts) if contacts is not None else []
    
    # For each agent, copy the body, actuator, and sensor, and append to the respective parent elements
    for i in range(0, num_agents):
        # Copy agent body and append
        new_agent_body = deepcopy(agent_body)
        for elem in new_agent_body.iter():
            if 'name' in elem.attrib:
                elem.set('name', f"{elem.get('name')}__{i}")
        root.find('worldbody').append(new_agent_body)

        # Copy actuators and append
        for actuator in original_actuators:
            new_actuator = deepcopy(actuator)
            if 'name' in new_actuator.attrib:
                new_actuator.set('name', f"{new_actuator.get('name')}__{i}")
            if 'joint' in new_actuator.attrib:
                new_actuator.set('joint', f"{new_actuator.get('joint')}__{i}")
            if 'jointinparent' in new_actuator.attrib:
                new_actuator.set('jointinparent', f"{new_actuator.get('jointinparent')}__{i}")
            if 'site' in new_actuator.attrib:
                new_actuator.set('site', f"{new_actuator.get('site')}__{i}")
            actuators.append(new_actuator)
                
        # Copy sensors and append
        for sensor in original_sensors:
            new_sensor = deepcopy(sensor)
            if 'name' in new_sensor.attrib:
                new_sensor.set('name', f"{new_sensor.get('name')}__{i}")
            if 'joint' in new_sensor.attrib:
                new_sensor.set('joint', f"{new_sensor.get('joint')}__{i}")
            if 'objname' in new_sensor.attrib:
                new_sensor.set('objname', f"{new_sensor.get('objname')}__{i}")
            if 'body' in new_sensor.attrib:
                new_sensor.set('body', f"{new_sensor.get('body')}__{i}")
            if 'site' in new_sensor.attrib:
                new_sensor.set('site', f"{new_sensor.get('site')}__{i}") 
            sensors.append(new_sensor)
        for contact in original_contact:
            new_contact = deepcopy(contact)
            if 'name' in new_contact.attrib:
                new_contact.set('name', f"{new_contact.get('name')}__{i}")
            if 'body1' in new_contact.attrib:
                new_contact.set('body1', f"{new_contact.get('body1')}__{i}")
            if 'body2' in new_contact.attrib:
                new_contact.set('body2', f"{new_contact.get('body2')}__{i}")
            contacts.append(new_contact)

        for equality in original_equality:
            new_equality = deepcopy(equality)
            if 'name' in new_equality.attrib:
                new_equality.set('name', f"{new_equality.get('name')}__{i}")
            if 'joint1' in new_equality.attrib:
                new_equality.set('joint1', f"{new_equality.get('joint1')}__{i}")
            if 'joint2' in new_equality.attrib:
                new_equality.set('joint2', f"{new_equality.get('joint2')}__{i}")
            equalitys.append(new_equality)

    for actuator in original_actuators:
        actuators.remove(actuator)
    for sensor in original_sensors:
        sensors.remove(sensor)
    for contact in original_contact:
        contacts.remove(contact)
    for equality in original_equality:
        equalitys.remove(equality)
    root.find('worldbody').remove(agent_body)
    # Return the modified XML content
    return ET.tostring(root, encoding="unicode")
