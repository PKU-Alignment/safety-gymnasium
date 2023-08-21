import json
import os
from random import randint, shuffle
from time import time

import numpy as np
import yaml
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from tqdm import tqdm

from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.hand_base.base_task import BaseTask


def quat_axis(q, axis=0):
    """??"""
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


class FreightFrankaCloseDrawer(BaseTask):
    def __init__(
        self,
        cfg,
        sim_params,
        physics_engine,
        device_type,
        device_id,
        headless,
        agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]],
        is_multi_agent=False,
        log_dir=None,
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.log_dir = log_dir
        self.up_axis = 'z'
        self.device_id = device_id if 'cuda' not in str(device_id) else int(device_id[-1])
        self.cfg['device_type'] = device_type
        self.cfg['device_id'] = self.device_id
        self.cfg['headless'] = headless
        self.device_type = device_type
        self.headless = headless
        self.device = 'cpu'
        self.use_handle = False
        if self.device_type == 'cuda' or self.device_type == 'GPU':
            self.device = 'cuda' + ':' + str(self.device_id)
        self.max_episode_length = self.cfg['env']['maxEpisodeLength']

        self.env_num_train = cfg['env']['numEnvs']
        self.env_num = self.env_num_train
        self.asset_root = os.path.dirname(os.path.abspath(__file__)).replace(
            'envs/tasks', 'envs/assets'
        )
        self.cabinet_num_train = cfg['env']['asset']['cabinetAssetNumTrain']
        self.cabinet_num = self.cabinet_num_train
        cabinet_train_list_len = len(cfg['env']['asset']['trainAssets'])
        self.cabinet_train_name_list = []
        self.exp_name = cfg['env']['env_name']
        print('Simulator: number of cabinets', self.cabinet_num)
        print('Simulator: number of environments', self.env_num)
        if self.cabinet_num_train:
            assert self.env_num_train % self.cabinet_num_train == 0

        assert (
            self.cabinet_num_train <= cabinet_train_list_len
        )  # the number of used length must less than real length
        assert self.env_num % self.cabinet_num == 0  # each cabinet should have equal number envs
        self.env_per_cabinet = self.env_num // self.cabinet_num
        self.task_meta = {
            'training_env_num': self.env_num_train,
            'need_update': True,
            'max_episode_length': self.max_episode_length,
            'obs_dim': cfg['env']['numObservations'],
        }
        for name in cfg['env']['asset']['trainAssets']:
            self.cabinet_train_name_list.append(cfg['env']['asset']['trainAssets'][name]['name'])

        self.cabinet_dof_lower_limits_tensor = torch.zeros(
            (self.cabinet_num, 1), device=self.device
        )
        self.cabinet_dof_upper_limits_tensor = torch.zeros(
            (self.cabinet_num, 1), device=self.device
        )
        self.cabinet_handle_pos_tensor = torch.zeros((self.cabinet_num, 3), device=self.device)
        self.cabinet_have_handle_tensor = torch.zeros((self.cabinet_num,), device=self.device)
        self.cabinet_open_dir_tensor = torch.zeros((self.cabinet_num,), device=self.device)
        self.cabinet_door_min_tensor = torch.zeros((self.cabinet_num, 3), device=self.device)
        self.cabinet_door_max_tensor = torch.zeros((self.cabinet_num, 3), device=self.device)

        self.env_ptr_list = []
        self.obj_loaded = False
        self.franka_loaded = False

        self.use_handle = cfg['task']['useHandle']
        self.use_stage = cfg['task']['useStage']
        self.use_slider = cfg['task']['useSlider']

        if 'useTaskId' in self.cfg['task'] and self.cfg['task']['useTaskId']:
            self.cfg['env']['numObservations'] += self.cabinet_num

        super().__init__(cfg=self.cfg, enable_camera_sensors=cfg['env']['enableCameraSensors'])
        # acquire tensors
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.rigid_body_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        )
        if (
            self.cfg['env']['driveMode'] == 'ik'
        ):  # inverse kinetic needs jacobian tensor, other drive mode don't need
            self.jacobian_tensor = gymtorch.wrap_tensor(
                self.gym.acquire_jacobian_tensor(self.sim, 'franka')
            )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)
        self.dof_state_tensor = self.dof_state_tensor.view(self.num_envs, -1, 2)
        self.rigid_body_tensor = self.rigid_body_tensor.view(self.num_envs, -1, 13)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        self.initial_dof_states = self.dof_state_tensor.clone()
        self.initial_root_states = self.root_tensor.clone()
        self.initial_rigid_body_states = self.rigid_body_tensor.clone()

        # precise slices of tensors
        env_ptr = self.env_ptr_list[0]
        franka1_actor = self.franka_actor_list[0]
        cabinet_actor = self.cabinet_actor_list[0]
        self.hand_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, franka1_actor, 'panda_hand', gymapi.DOMAIN_ENV
        )
        self.freight_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, franka1_actor, 'base_link', gymapi.DOMAIN_ENV
        )
        self.hand_lfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, franka1_actor, 'panda_leftfinger', gymapi.DOMAIN_ENV
        )
        self.hand_rfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, franka1_actor, 'panda_rightfinger', gymapi.DOMAIN_ENV
        )
        self.cabinet_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, cabinet_actor, self.cabinet_rig_name, gymapi.DOMAIN_ENV
        )
        self.cabinet_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, cabinet_actor, self.cabinet_base_rig_name, gymapi.DOMAIN_ENV
        )
        self.cabinet_dof_index = self.gym.find_actor_dof_index(
            env_ptr, cabinet_actor, self.cabinet_dof_name, gymapi.DOMAIN_ENV
        )

        self.hand_rigid_body_tensor = self.rigid_body_tensor[:, self.hand_rigid_body_index, :]
        self.franka_dof_tensor = self.dof_state_tensor[:, : self.franka_num_dofs, :]
        self.cabinet_dof_tensor = self.dof_state_tensor[:, self.cabinet_dof_index, :]
        self.cabinet_dof_tensor_spec = self._detailed_view(self.cabinet_dof_tensor)
        self.cabinet_door_rigid_body_tensor = self.rigid_body_tensor[
            :, self.cabinet_rigid_body_index, :
        ]
        self.franka_root_tensor = self.root_tensor[:, 0, :]
        self.cabinet_root_tensor = self.root_tensor[:, 1, :]

        self.cabinet_dof_target = self.initial_dof_states[:, self.cabinet_dof_index, 0]
        self.dof_dim = self.franka_num_dofs + 1
        self.pos_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.eff_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.stage = torch.zeros((self.num_envs), device=self.device)

        # initialization of pose
        self.cabinet_dof_coef = -1.0
        self.success_dof_states = self.cabinet_dof_lower_limits_tensor[:, 0].clone()
        self.initial_dof_states.view(self.cabinet_num, self.env_per_cabinet, -1, 2)[
            :, :, self.cabinet_dof_index, 0
        ] = (torch.ones((self.cabinet_num, 1), device=self.device) * 0.2)

        self.map_dis_bar = cfg['env']['map_dis_bar']
        self.action_speed_scale = cfg['env']['actionSpeedScale']

        # params of randomization
        self.cabinet_reset_position_noise = cfg['env']['reset']['cabinet']['resetPositionNoise']
        self.cabinet_reset_rotation_noise = cfg['env']['reset']['cabinet']['resetRotationNoise']
        self.cabinet_reset_dof_pos_interval = cfg['env']['reset']['cabinet'][
            'resetDofPosRandomInterval'
        ]
        self.cabinet_reset_dof_vel_interval = cfg['env']['reset']['cabinet'][
            'resetDofVelRandomInterval'
        ]
        self.franka_reset_position_noise = cfg['env']['reset']['franka']['resetPositionNoise']
        self.franka_reset_rotation_noise = cfg['env']['reset']['franka']['resetRotationNoise']
        self.franka_reset_dof_pos_interval = cfg['env']['reset']['franka'][
            'resetDofPosRandomInterval'
        ]
        self.franka_reset_dof_vel_interval = cfg['env']['reset']['franka'][
            'resetDofVelRandomInterval'
        ]

        # params for success rate
        self.success = torch.zeros((self.env_num,), device=self.device)
        self.success_rate = torch.zeros((self.env_num,), device=self.device)
        self.success_buf = torch.zeros((self.env_num,), device=self.device).long()

        self.average_reward = None

        # flags for switching between training and evaluation mode
        self.train_mode = True

        self.num_freight_obs = 3 * 2
        self.num_franka_obs = 9 * 2

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        self._create_ground_plane()
        self._place_agents(self.cfg['env']['numEnvs'], self.cfg['env']['envSpacing'])

    def _franka_init_pose(self):
        initial_franka_pose = gymapi.Transform()

        initial_franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        if self.cfg['task']['target'] == 'close':
            self.cabinet_dof_coef = -1.0
            self.success_dof_states = self.cabinet_dof_lower_limits_tensor[:, 0].clone()
            initial_franka_pose.p = gymapi.Vec3(0.6, 0.0, 0.02)

        return initial_franka_pose

    def _cam_pose(self):
        cam_pos = gymapi.Vec3(13.0, 13.0, 6.0)
        cam_target = gymapi.Vec3(8.0, 8.0, 0.1)

        return cam_pos, cam_target

    def _load_franka(self, env_ptr, env_id):
        if self.franka_loaded == False:
            self.franka_actor_list = []

            asset_root = self.asset_root
            asset_file = 'urdf/freight_franka/urdf/freight_franka.urdf'
            self.gripper_length = 0.13
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
            asset_options.flip_visual_attachments = True
            asset_options.armature = 0.01
            self.franka_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

            self.franka_loaded = True

        (
            franka_dof_max_torque,
            franka_dof_lower_limits,
            franka_dof_upper_limits,
        ) = self._get_dof_property(self.franka_asset)
        self.franka_dof_max_torque_tensor = torch.tensor(franka_dof_max_torque, device=self.device)
        self.franka_dof_mean_limits_tensor = torch.tensor(
            (franka_dof_lower_limits + franka_dof_upper_limits) / 2, device=self.device
        )
        self.franka_dof_limits_range_tensor = torch.tensor(
            (franka_dof_upper_limits - franka_dof_lower_limits) / 2, device=self.device
        )
        self.franka_dof_lower_limits_tensor = torch.tensor(
            franka_dof_lower_limits, device=self.device
        )
        self.franka_dof_upper_limits_tensor = torch.tensor(
            franka_dof_upper_limits, device=self.device
        )

        dof_props = self.gym.get_asset_dof_properties(self.franka_asset)

        # use position drive for all dofs
        dof_props['driveMode'][:-2].fill(gymapi.DOF_MODE_POS)
        dof_props['stiffness'][:-2].fill(400.0)
        dof_props['damping'][:-2].fill(40.0)
        # grippers
        dof_props['driveMode'][-2:].fill(gymapi.DOF_MODE_EFFORT)
        dof_props['stiffness'][-2:].fill(0.0)
        dof_props['damping'][-2:].fill(0.0)

        # root pose
        initial_franka_pose = self._franka_init_pose()

        # set start dof
        self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
        default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        default_dof_pos[:-2] = (franka_dof_lower_limits + franka_dof_upper_limits)[:-2] * 0.3
        # grippers open
        default_dof_pos[-2:] = franka_dof_upper_limits[-2:]
        franka_dof_state = np.zeros_like(franka_dof_max_torque, gymapi.DofState.dtype)
        franka_dof_state['pos'] = default_dof_pos

        franka_actor = self.gym.create_actor(
            env_ptr, self.franka_asset, initial_franka_pose, 'franka', env_id, 1, 0
        )

        self.gym.set_actor_dof_properties(env_ptr, franka_actor, dof_props)
        self.gym.set_actor_dof_states(env_ptr, franka_actor, franka_dof_state, gymapi.STATE_ALL)
        self.franka_actor_list.append(franka_actor)

    def _get_dof_property(self, asset):
        dof_props = self.gym.get_asset_dof_properties(asset)
        dof_num = self.gym.get_asset_dof_count(asset)
        dof_lower_limits = []
        dof_upper_limits = []
        dof_max_torque = []
        for i in range(dof_num):
            dof_max_torque.append(dof_props['effort'][i])
            dof_lower_limits.append(dof_props['lower'][i])
            dof_upper_limits.append(dof_props['upper'][i])
        dof_max_torque = np.array(dof_max_torque)
        dof_lower_limits = np.array(dof_lower_limits)
        dof_upper_limits = np.array(dof_upper_limits)
        return dof_max_torque, dof_lower_limits, dof_upper_limits

    def _obj_init_pose(self, min_dict, max_dict):
        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(-max_dict[2], 0.0, -min_dict[1] + 0.3)
        cabinet_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        return cabinet_start_pose

    def _load_obj_asset(self):
        self.cabinet_asset_name_list = []
        self.cabinet_asset_list = []
        self.cabinet_pose_list = []
        self.cabinet_actor_list = []
        self.cabinet_pc = []

        train_len = len(self.cfg['env']['asset']['trainAssets'].items())
        train_len = min(train_len, self.cabinet_num_train)
        total_len = train_len
        used_len = min(total_len, self.cabinet_num)

        random_asset = self.cfg['env']['asset']['randomAsset']
        select_train_asset = [i for i in range(train_len)]
        if (
            random_asset
        ):  # if we need random asset list from given dataset, we shuffle the list to be read
            shuffle(select_train_asset)
        select_train_asset = select_train_asset[:train_len]

        with tqdm(total=used_len) as pbar:
            pbar.set_description('Loading cabinet assets:')
            cur = 0

            asset_list = []

            # prepare the assets to be used
            for id, (name, val) in enumerate(self.cfg['env']['asset']['trainAssets'].items()):
                if id in select_train_asset:
                    asset_list.append((id, (name, val)))

            for id, (name, val) in asset_list:
                self.cabinet_asset_name_list.append(name)

                asset_options = gymapi.AssetOptions()
                asset_options.fix_base_link = True
                asset_options.disable_gravity = True
                asset_options.collapse_fixed_joints = True
                asset_options.use_mesh_materials = True
                asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                asset_options.override_com = True
                asset_options.override_inertia = True
                asset_options.vhacd_enabled = True
                asset_options.vhacd_params = gymapi.VhacdParams()
                asset_options.vhacd_params.resolution = 512

                cabinet_asset = self.gym.load_asset(
                    self.sim, self.asset_root, val['path'], asset_options
                )
                self.cabinet_asset_list.append(cabinet_asset)

                with open(os.path.join(self.asset_root, val['boundingBox'])) as f:
                    cabinet_bounding_box = json.load(f)
                    min_dict = cabinet_bounding_box['min']
                    max_dict = cabinet_bounding_box['max']

                dof_dict = self.gym.get_asset_dof_dict(cabinet_asset)
                if len(dof_dict) != 1:
                    print(val['path'])
                    print(len(dof_dict))
                assert len(dof_dict) == 1
                self.cabinet_dof_name = list(dof_dict.keys())[0]

                rig_dict = self.gym.get_asset_rigid_body_dict(cabinet_asset)
                assert len(rig_dict) == 2
                self.cabinet_rig_name = list(rig_dict.keys())[1]
                self.cabinet_base_rig_name = list(rig_dict.keys())[0]
                assert self.cabinet_rig_name != 'base'

                self.cabinet_pose_list.append(self._obj_init_pose(min_dict, max_dict))

                max_torque, lower_limits, upper_limits = self._get_dof_property(cabinet_asset)
                self.cabinet_dof_lower_limits_tensor[cur, :] = torch.tensor(
                    lower_limits[0], device=self.device
                )
                self.cabinet_dof_upper_limits_tensor[cur, :] = torch.tensor(
                    upper_limits[0], device=self.device
                )

                dataset_path = self.cfg['env']['asset']['datasetPath']

                with open(os.path.join(self.asset_root, dataset_path, name, 'handle.yaml')) as f:
                    handle_dict = yaml.safe_load(f)
                    self.cabinet_have_handle_tensor[cur] = handle_dict['has_handle']
                    self.cabinet_handle_pos_tensor[cur][0] = handle_dict['pos']['x']
                    self.cabinet_handle_pos_tensor[cur][1] = handle_dict['pos']['y']
                    self.cabinet_handle_pos_tensor[cur][2] = handle_dict['pos']['z']

                with open(os.path.join(self.asset_root, dataset_path, name, 'door.yaml')) as f:
                    door_dict = yaml.safe_load(f)
                    if (door_dict['open_dir'] not in [-1, 1]) and (
                        not self.cfg['task']['useDrawer']
                    ):
                        print(
                            'Warning: Door type of {} is not supported, possibly a unrecognized open direction',
                            name,
                        )
                    if self.cfg['task']['useDrawer']:
                        self.cabinet_open_dir_tensor[cur] = 1
                    else:
                        self.cabinet_open_dir_tensor[cur] = door_dict['open_dir']
                    self.cabinet_door_min_tensor[cur][0] = door_dict['bounding_box']['xmin']
                    self.cabinet_door_min_tensor[cur][1] = door_dict['bounding_box']['ymin']
                    self.cabinet_door_min_tensor[cur][2] = door_dict['bounding_box']['zmin']
                    self.cabinet_door_max_tensor[cur][0] = door_dict['bounding_box']['xmax']
                    self.cabinet_door_max_tensor[cur][1] = door_dict['bounding_box']['ymax']
                    self.cabinet_door_max_tensor[cur][2] = door_dict['bounding_box']['zmax']

                pbar.update(1)
                cur += 1
        # flag for hazard which is forbidden to enter
        hazard_asset_options = gymapi.AssetOptions()
        hazard_asset_options.fix_base_link = True
        hazard_asset_options.disable_gravity = False
        self.hazard_asset = self.gym.create_box(self.sim, 0.5, 1.0, 0.01, hazard_asset_options)
        self.hazard_pose = gymapi.Transform()
        self.hazard_pose.p = gymapi.Vec3(0.05, 0.0, 0.005)  # franka:0, 0.0, 0
        self.hazard_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    def _load_obj(self, env_ptr, env_id):
        if self.obj_loaded == False:
            self._load_obj_asset()

            self.cabinet_handle_pos_tensor = self.cabinet_handle_pos_tensor.repeat_interleave(
                self.env_per_cabinet, dim=0
            )
            self.cabinet_have_handle_tensor = self.cabinet_have_handle_tensor.repeat_interleave(
                self.env_per_cabinet, dim=0
            )
            self.cabinet_open_dir_tensor = self.cabinet_open_dir_tensor.repeat_interleave(
                self.env_per_cabinet, dim=0
            )
            self.cabinet_door_min_tensor = self.cabinet_door_min_tensor.repeat_interleave(
                self.env_per_cabinet, dim=0
            )
            self.cabinet_door_max_tensor = self.cabinet_door_max_tensor.repeat_interleave(
                self.env_per_cabinet, dim=0
            )

            self.cabinet_door_edge_min_l = torch.zeros_like(self.cabinet_door_min_tensor)
            self.cabinet_door_edge_max_l = torch.zeros_like(self.cabinet_door_max_tensor)
            self.cabinet_door_edge_min_r = torch.zeros_like(self.cabinet_door_min_tensor)
            self.cabinet_door_edge_max_r = torch.zeros_like(self.cabinet_door_max_tensor)
            self.cabinet_door_edge_min = torch.zeros_like(self.cabinet_door_min_tensor)
            self.cabinet_door_edge_max = torch.zeros_like(self.cabinet_door_max_tensor)
            self.cabinet_door_edge_min_l[:, 0] = self.cabinet_door_max_tensor[:, 0]
            self.cabinet_door_edge_min_l[:, 1] = self.cabinet_door_min_tensor[:, 1]
            self.cabinet_door_edge_min_l[:, 2] = self.cabinet_door_min_tensor[:, 2]
            self.cabinet_door_edge_max_l[:, 0] = self.cabinet_door_max_tensor[:, 0]
            self.cabinet_door_edge_max_l[:, 1] = self.cabinet_door_max_tensor[:, 1]
            self.cabinet_door_edge_max_l[:, 2] = self.cabinet_door_min_tensor[:, 2]

            self.cabinet_door_edge_min_r[:, 0] = self.cabinet_door_min_tensor[:, 0]
            self.cabinet_door_edge_min_r[:, 1] = self.cabinet_door_min_tensor[:, 1]
            self.cabinet_door_edge_min_r[:, 2] = self.cabinet_door_min_tensor[:, 2]
            self.cabinet_door_edge_max_r[:, 0] = self.cabinet_door_min_tensor[:, 0]
            self.cabinet_door_edge_max_r[:, 1] = self.cabinet_door_max_tensor[:, 1]
            self.cabinet_door_edge_max_r[:, 2] = self.cabinet_door_min_tensor[:, 2]
            self.cabinet_door_edge_max = torch.where(
                self.cabinet_open_dir_tensor.view(self.num_envs, -1) < -0.5,
                self.cabinet_door_edge_max_l,
                self.cabinet_door_edge_max_r,
            )
            self.cabinet_door_edge_min = torch.where(
                self.cabinet_open_dir_tensor.view(self.num_envs, -1) < -0.5,
                self.cabinet_door_edge_min_l,
                self.cabinet_door_edge_min_r,
            )

            self.obj_loaded = True

        obj_actor = self.gym.create_actor(
            env_ptr,
            self.cabinet_asset_list[0],
            self.cabinet_pose_list[0],
            f'cabinet{env_id}',
            env_id,
            2,
            0,
        )
        cabinet_dof_props = self.gym.get_asset_dof_properties(self.cabinet_asset_list[0])
        cabinet_dof_props['stiffness'][0] = 30.0
        cabinet_dof_props['friction'][0] = 2.0
        cabinet_dof_props['effort'][0] = 4.0
        cabinet_dof_props['driveMode'][0] = gymapi.DOF_MODE_POS
        self.gym.set_actor_dof_properties(env_ptr, obj_actor, cabinet_dof_props)
        self.cabinet_actor_list.append(obj_actor)

        hazard_actor = self.gym.create_actor(
            env_ptr,
            self.hazard_asset,
            self.hazard_pose,
            f'hazard-{env_id}',
            env_id,  # collision group
            3,  # filter
            0,
        )
        # set hazard area as red
        self.gym.set_rigid_body_color(
            env_ptr,
            hazard_actor,
            self.gym.find_asset_rigid_body_index(self.hazard_asset, 'box'),
            gymapi.MESH_VISUAL_AND_COLLISION,
            gymapi.Vec3(1.0, 0.0, 0.0),
        )

    def _place_agents(self, env_num, spacing):
        print('Simulator: creating agents')

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.space_middle = torch.zeros((env_num, 3), device=self.device)
        self.space_range = torch.zeros((env_num, 3), device=self.device)
        self.space_middle[:, 0] = self.space_middle[:, 1] = 0
        self.space_middle[:, 2] = spacing / 2
        self.space_range[:, 0] = self.space_range[:, 1] = spacing
        self.space_middle[:, 2] = spacing / 2
        num_per_row = int(np.sqrt(env_num))

        with tqdm(total=env_num) as pbar:
            pbar.set_description('Enumerating envs:')
            for env_id in range(env_num):
                env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
                self.env_ptr_list.append(env_ptr)
                self._load_franka(env_ptr, env_id)
                self._load_obj(env_ptr, env_id)
                pbar.update(1)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 0.1
        plane_params.dynamic_friction = 0.1
        self.gym.add_ground(self.sim, plane_params)

    def _get_reward_cost_done(self):
        door_pos = self.cabinet_door_rigid_body_tensor[:, :3]
        door_rot = self.cabinet_door_rigid_body_tensor[:, 3:7]
        hand_rot = self.hand_rigid_body_tensor[..., 3:7]
        hand_down_dir = quat_axis(hand_rot, 2)
        hand_grip_dir = quat_axis(hand_rot, 1)
        hand_sep_dir = quat_axis(hand_rot, 0)
        handle_pos = quat_apply(door_rot, self.cabinet_handle_pos_tensor) + door_pos
        handle_x = quat_axis(door_rot, 0) * self.cabinet_open_dir_tensor.view(-1, 1)
        handle_z = quat_axis(door_rot, 1)

        cabinet_door_relative_o = door_pos + quat_apply(door_rot, self.cabinet_door_edge_min)
        cabinet_door_relative_x = -handle_x
        cabinet_door_relative_y = -quat_axis(door_rot, 2)
        cabinet_door_relative_z = quat_axis(door_rot, 1)

        franka_rfinger_pos = (
            self.rigid_body_tensor[:, self.hand_lfinger_rigid_body_index][:, 0:3]
            + hand_down_dir * 0.075
        )
        franka_lfinger_pos = (
            self.rigid_body_tensor[:, self.hand_rfinger_rigid_body_index][:, 0:3]
            + hand_down_dir * 0.075
        )

        freight_pos = self.rigid_body_tensor[:, self.freight_rigid_body_index][:, 0:3]

        # close door or drawer
        door_reward = self.cabinet_dof_coef * self.cabinet_dof_tensor[:, 0]
        action_penalty = torch.sum(
            (self.pos_act[:, :7] - self.franka_dof_tensor[:, :7, 0]) ** 2, dim=-1
        )

        d = torch.norm(self.hand_tip_pos - handle_pos, p=2, dim=-1)

        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.1, dist_reward * 2, dist_reward)
        dist_reward *= self.cabinet_have_handle_tensor

        dot1 = (hand_grip_dir * handle_z).sum(dim=-1)
        dot2 = (-hand_sep_dir * handle_x).sum(dim=-1)
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        diff_from_success = torch.abs(
            self.cabinet_dof_tensor_spec[:, :, 0]
            - self.success_dof_states.view(self.cabinet_num, -1)
        ).view(-1)
        success = diff_from_success < 0.01
        success_bonus = success

        open_reward = self.cabinet_dof_tensor[:, 0] * 10

        self.rew_buf = 1.0 * dist_reward + 0.5 * rot_reward - 1 * open_reward

        self.cost_x_range = torch.tensor([-0.2, 0.3])
        self.cost_y_range = torch.tensor([-0.5, 0.5])

        freight_x = freight_pos[:, 0]
        freight_y = freight_pos[:, 1]

        within_x = (self.cost_x_range[0] <= freight_x) & (freight_x <= self.cost_x_range[1])
        within_y = (self.cost_y_range[0] <= freight_y) & (freight_y <= self.cost_y_range[1])

        self.cost_buf = (within_x & within_y).type(torch.float32)

        # set the type

        time_out = self.progress_buf >= self.max_episode_length
        self.reset_buf = self.reset_buf | time_out
        self.success_buf = self.success_buf | success
        self.success = self.success_buf & time_out

        old_coef = 1.0 - time_out * 0.1
        new_coef = time_out * 0.1

        self.success_rate = self.success_rate * old_coef + success * new_coef

        return self.rew_buf, self.cost_buf, self.reset_buf

    def _get_base_observation(self, suggested_gt=None):
        hand_rot = self.hand_rigid_body_tensor[..., 3:7]
        hand_down_dir = quat_axis(hand_rot, 2)
        self.hand_tip_pos = (
            self.hand_rigid_body_tensor[..., 0:3] + hand_down_dir * self.gripper_length
        )  # calculating middle of two fingers
        self.hand_rot = hand_rot

        dim = 57

        state = torch.zeros((self.num_envs, dim), device=self.device)

        joints = self.franka_num_dofs
        # joint dof value
        state[:, :joints].copy_(
            (
                2
                * (
                    self.franka_dof_tensor[:, :joints, 0]
                    - self.franka_dof_lower_limits_tensor[:joints]
                )
                / (
                    self.franka_dof_upper_limits_tensor[:joints]
                    - self.franka_dof_lower_limits_tensor[:joints]
                )
            )
            - 1
        )
        # joint dof velocity
        state[:, joints : joints * 2].copy_(self.franka_dof_tensor[:, :joints, 1])
        # cabinet dof
        state[:, joints * 2 : joints * 2 + 2].copy_(self.cabinet_dof_tensor)
        # hand
        state[:, joints * 2 + 2 : joints * 2 + 15].copy_(
            relative_pose(self.franka_root_tensor, self.hand_rigid_body_tensor).view(
                self.env_num, -1
            )
        )
        # actions
        state[:, joints * 2 + 15 : joints * 3 + 15].copy_(self.actions[:, :joints])

        state[:, joints * 3 + 15 : joints * 3 + 15 + 3].copy_(
            self.franka_root_tensor[:, 0:3] - self.cabinet_handle_pos_tensor
        )
        state[:, joints * 3 + 15 + 3 : joints * 3 + 15 + 3 + 3].copy_(
            self.cabinet_handle_pos_tensor - self.hand_tip_pos
        )

        return state

    def _refresh_observation(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.obs_buf = self._get_base_observation()

    def _perform_actions(self, actions):
        actions = actions.to(self.device)
        self.actions = actions

        joints = self.franka_num_dofs - 2
        self.pos_act[:, :-3] = (
            self.pos_act[:, :-3] + actions[:, 0:joints] * self.dt * self.action_speed_scale
        )
        self.pos_act[:, :joints] = tensor_clamp(
            self.pos_act[:, :joints],
            self.franka_dof_lower_limits_tensor[:joints],
            self.franka_dof_upper_limits_tensor[:joints],
        )

        self.eff_act[:, -3] = actions[:, -2] * self.franka_dof_max_torque_tensor[-2]  # gripper1
        self.eff_act[:, -2] = actions[:, -1] * self.franka_dof_max_torque_tensor[-1]  # gripper2
        self.pos_act[:, self.franka_num_dofs] = self.cabinet_dof_target  # door reverse force
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.pos_act.view(-1))
        )
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self.eff_act.view(-1))
        )

    def _draw_line(self, src, dst):
        line_vec = np.stack([src.cpu().numpy(), dst.cpu().numpy()]).flatten().astype(np.float32)
        color = np.array([1, 0, 0], dtype=np.float32)
        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(self.viewer, self.env_ptr_list[0], self.env_num, line_vec, color)

    # @TimeCounter
    def step(self, actions):
        self._perform_actions(actions)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        if not self.headless:
            self.render()
        if self.cfg['env']['enableCameraSensors'] == True:
            self.gym.step_graphics(self.sim)

        self.progress_buf += 1

        self._refresh_observation()

        reward, cost, done = self._get_reward_cost_done()

        done = self.reset_buf.clone()
        success = self.success.clone()
        self._partial_reset(self.reset_buf)

        if self.average_reward == None:
            self.average_reward = self.rew_buf.mean()
        else:
            self.average_reward = self.rew_buf.mean() * 0.01 + self.average_reward * 0.99
        self.extras['successes'] = success
        self.extras['success_rate'] = self.success_rate
        return self.obs_buf, self.rew_buf, self.cost_buf, done, None

    def _partial_reset(self, to_reset='all'):
        """
        reset those need to be reseted
        """

        if to_reset == 'all':
            to_reset = np.ones((self.env_num,))
        reseted = False
        for env_id, reset in enumerate(to_reset):
            # is reset:
            if reset.item():
                # need randomization
                reset_dof_states = self.initial_dof_states[env_id].clone()
                reset_root_states = self.initial_root_states[env_id].clone()
                franka_reset_pos_tensor = reset_root_states[0, :3]
                franka_reset_rot_tensor = reset_root_states[0, 3:7]
                franka_reset_dof_pos_tensor = reset_dof_states[: self.franka_num_dofs, 0]
                franka_reset_dof_vel_tensor = reset_dof_states[: self.franka_num_dofs, 1]
                cabinet_reset_pos_tensor = reset_root_states[1, :3]
                cabinet_reset_rot_tensor = reset_root_states[1, 3:7]
                cabinet_reset_dof_pos_tensor = reset_dof_states[self.franka_num_dofs :, 0]
                cabinet_reset_dof_vel_tensor = reset_dof_states[self.franka_num_dofs :, 1]

                cabinet_type = env_id // self.env_per_cabinet

                self.intervaledRandom_(franka_reset_pos_tensor, self.franka_reset_position_noise)
                self.intervaledRandom_(franka_reset_rot_tensor, self.franka_reset_rotation_noise)
                self.intervaledRandom_(
                    franka_reset_dof_pos_tensor,
                    self.franka_reset_dof_pos_interval,
                    self.franka_dof_lower_limits_tensor,
                    self.franka_dof_upper_limits_tensor,
                )
                self.intervaledRandom_(
                    franka_reset_dof_vel_tensor, self.franka_reset_dof_vel_interval
                )
                self.intervaledRandom_(cabinet_reset_pos_tensor, self.cabinet_reset_position_noise)
                self.intervaledRandom_(cabinet_reset_rot_tensor, self.cabinet_reset_rotation_noise)
                self.intervaledRandom_(
                    cabinet_reset_dof_pos_tensor,
                    self.cabinet_reset_dof_pos_interval,
                    self.cabinet_dof_lower_limits_tensor[cabinet_type],
                    self.cabinet_dof_upper_limits_tensor[cabinet_type],
                )
                self.intervaledRandom_(
                    cabinet_reset_dof_vel_tensor, self.cabinet_reset_dof_vel_interval
                )

                self.dof_state_tensor[env_id].copy_(reset_dof_states)
                self.root_tensor[env_id].copy_(reset_root_states)
                reseted = True
                self.progress_buf[env_id] = 0
                self.reset_buf[env_id] = 0
                self.success_buf[env_id] = 0

        if reseted:
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state_tensor))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))

    def reset(self, to_reset='all'):
        self._partial_reset(to_reset)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if not self.headless:
            self.render()
        if self.cfg['env']['enableCameraSensors'] == True:
            self.gym.step_graphics(self.sim)

        self._refresh_observation()
        success = self.success.clone()
        reward, cost, done = self._get_reward_cost_done()

        self.extras['successes'] = success
        self.extras['success_rate'] = self.success_rate
        return self.obs_buf, self.rew_buf, self.cost_buf, self.reset_buf, None

    def save(self, path, iteration):
        pass

    def update(self, it=0):
        pass

    def train(self):  # changing mode to train
        self.train_mode = True

    def eval(self):  # changing mode to eval
        self.train_mode = False

    def _get_cabinet_door_mask(self):
        door_pos = self.cabinet_door_rigid_body_tensor[:, :3]
        door_rot = self.cabinet_door_rigid_body_tensor[:, 3:7]

        handle_x = quat_axis(door_rot, 0) * self.cabinet_open_dir_tensor.view(-1, 1)

        cabinet_door_edge_min = door_pos + quat_apply(door_rot, self.cabinet_door_edge_min)

        cabinet_door_relative_o = cabinet_door_edge_min
        cabinet_door_relative_x = handle_x
        cabinet_door_relative_y = quat_axis(door_rot, 2)
        cabinet_door_relative_z = quat_axis(door_rot, 1)

        cabinet_door_x = self.cabinet_door_max_tensor[:, 0] - self.cabinet_door_min_tensor[:, 0]
        cabinet_door_y = self.cabinet_door_max_tensor[:, 2] - self.cabinet_door_min_tensor[:, 2]
        cabinet_door_z = self.cabinet_door_max_tensor[:, 1] - self.cabinet_door_min_tensor[:, 1]

        return (
            cabinet_door_relative_o,  # origin coord
            cabinet_door_relative_x,  # normalized x axis
            cabinet_door_relative_y,
            cabinet_door_relative_z,
            cabinet_door_x,  # length of x axis
            cabinet_door_y,
            cabinet_door_z,
        )

    def _detailed_view(self, tensor):
        shape = tensor.shape
        return tensor.view(self.cabinet_num, self.env_per_cabinet, *shape[1:])

    def intervaledRandom_(self, tensor, dist, lower=None, upper=None):
        tensor += torch.rand(tensor.shape, device=self.device) * dist * 2 - dist
        if lower is not None and upper is not None:
            torch.clamp_(tensor, min=lower, max=upper)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(j_eef, device, dpose, num_envs):
    # Set controller parameters
    # IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u


def relative_pose(src, dst):
    shape = dst.shape
    p = dst.view(-1, shape[-1])[:, :3] - src.view(-1, src.shape[-1])[:, :3]
    ip = dst.view(-1, shape[-1])[:, 3:]
    ret = torch.cat((p, ip), dim=1)
    return ret.view(*shape)
