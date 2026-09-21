# ==============================================================================
# [Bench2Drive-Style] MODIFIED FILE -- script
# Author: YinXuan1223   (carried over from the ORION_NYCUACM_version fork)
#
# The CARLA leaderboard agent, extended for style-conditioned closed-loop driving:
#   * driving style is injected per step as input_data_batch['driving_style'].
#     FORCE_DRIVING_STYLE in [-1, 1] sets it; with STYLE_PER_REP=1 the style is
#     instead taken from the leaderboard repetition index (rep0/1/2 ->
#     aggressive/neutral/conservative), so --repetitions=3 drives each route in all
#     three styles back to back. STYLE_GUIDANCE is read by the model for CFG.
#   * cfg.half_precision_load -> cast to cfg.half_precision_dtype ('bfloat16'
#     default, 'float16' for the fp16 config) before .cuda(), so the 7.5B model
#     fits a 24 GB card.
#   * imports RoutePlanner from this repo (StyleOrion.team_code.planner) instead of
#     Bench2DriveZoo -- the local one returns two route nodes, which is what tick()
#     unpacks. Like the Bench2DriveZoo import above it, this resolves as a namespace
#     package, so the PARENT of StyleOrion/ must be on PYTHONPATH.
#   * saving trimmed to the front camera at 10 Hz (metric_info.json is still
#     written, it is required for the driving-score benchmark).
#   * STYLE_VIEW=1 writes an annotated live view (style, speed, controls and a
#     top-down plan mini-map) to /dev/shm for VNC; it is wrapped so a rendering
#     error can never affect driving.
# ==============================================================================
# ------------------------------------------------------------------------
# Modified from Bench2Drive(https://github.com/Thinklab-SJTU/Bench2Drive)
# Copyright (c) Xiaomi, Inc. All rights reserved.
# ------------------------------------------------------------------------

import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
from Bench2DriveZoo.team_code.pid_controller import PIDController
# Use ORION's local planner: its run_step returns TWO nodes (route[0], route[1]),
# which tick() unpacks as ((_, curr_command), (near_node, near_command)).
# Bench2DriveZoo's planner returns a single node -> "cannot unpack RoadOption".
from StyleOrion.team_code.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent
from mmcv import Config
from mmcv.models import build_model 
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint,wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from pyquaternion import Quaternion
from scipy.optimize import fsolve
import re
SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
# [StyleDrive] When STYLE_PER_REP=1, the driving style is chosen from the leaderboard
# repetition index (parsed from save_name's "_rep{k}_") instead of the FORCE_DRIVING_STYLE
# env. Run with --repetitions=3 so each route is driven rep0/rep1/rep2 back-to-back =
# aggressive / neutral / conservative, before moving to the next route.
STYLE_PER_REP = os.environ.get('STYLE_PER_REP', '0') == '1'
REP_STYLE_MAP = {0: 1.0, 1: 0.0, 2: -1.0}   # rep0 aggressive / rep1 neutral / rep2 conservative

def get_entry_point():
    return 'OrionAgent'

class OrionAgent(autonomous_agent.AutonomousAgent):

    def setup_model(self, model, pipeline):
        self.model = model
        self.inference_only_pipeline = pipeline

    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = 0
        self.pidcontroller = PIDController() 
        self.config_path = path_to_conf_file.split('+')[0]
        self.ckpt_path = path_to_conf_file.split('+')[1]
        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
        else:
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        # [StyleDrive] resolve the driving style for THIS route once (constant for the route).
        self.style_val = float(os.environ.get('FORCE_DRIVING_STYLE', '0.0'))
        if STYLE_PER_REP:
            m = re.search(r'_rep(\d+)_', self.save_name)
            rep = int(m.group(1)) if m else 0
            self.style_val = REP_STYLE_MAP.get(rep, 0.0)
            print('[StyleDrive] STYLE_PER_REP: rep=%d -> driving_style=%+.1f (%s)' % (
                rep, self.style_val,
                {1.0: 'aggressive', 0.0: 'neutral', -1.0: 'conservative'}.get(self.style_val, 'custom')), flush=True)
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        if not (hasattr(self, 'model') and self.model is not None and 
                hasattr(self, 'inference_only_pipeline') and self.inference_only_pipeline is not None):
            cfg = Config.fromfile(self.config_path)
            if hasattr(cfg, 'plugin'):
                if cfg.plugin:
                    import importlib
                    if hasattr(cfg, 'plugin_dir'):
                        plugin_dir = cfg.plugin_dir
                        plugin_dir = os.path.join("Bench2DriveZoo", plugin_dir)
                        _module_dir = os.path.dirname(plugin_dir)
                        _module_dir = _module_dir.split('/')
                        _module_path = _module_dir[0]
                        for m in _module_dir[1:]:
                            _module_path = _module_path + '.' + m
                        print(_module_path)
                        plg_lib = importlib.import_module(_module_path)  
    
            self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
            checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu')
            # [StyleDrive] cast to half precision before .cuda() so the 7.5B model fits a 24G
            # card; fp32 weights would OOM on a 3090. dtype is configurable: 'bfloat16' (default,
            # the diff_style/baseline path) or 'float16' (the authors' validated FP16 path, which
            # forward() then autocasts to fp16 to match). fp16 has a 10-bit mantissa vs bf16's 7,
            # so it preserves the generative planner's trajectory precision.
            if cfg.get('half_precision_load', False):
                _hp_dtype = getattr(torch, cfg.get('half_precision_dtype', 'bfloat16'))
                self.model = self.model.to(_hp_dtype)
            self.model.cuda()
            self.model.eval()
            self.inference_only_pipeline = []
            for inference_only_pipeline in cfg.inference_only_pipeline:
                if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph']:
                    self.inference_only_pipeline.append(inference_only_pipeline)
            self.inference_only_pipeline = Compose(self.inference_only_pipeline)

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.lat_ref, self.lon_ref = 42.0, 2.0
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control = control
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            # string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string = self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            # [StyleDrive] only the front camera is saved (see save()); 10 Hz sampling.
            (self.save_path / 'rgb_front').mkdir()
   
        # write extrinsics directly
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                                  [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                                  [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                       [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
         'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                                 [ 0.  ,  0.  , -1.  , -0.24],
                                 [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                      [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
          'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0        
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}
  
  

    def sensors(self):
        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },       
            ]
        if IS_BENCH2DRIVE:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            # img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            img = input_data[cam][1][:, :, :3]
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        
        #NOTE@Jianfeng: we directly use BGR image and let pipeline do the convert
        # breakpoint()
        # cv2.imwrite('./work_dirs/tick_input_img.jpg', img)
        # bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        bev = input_data['bev'][1][:, :, :3]
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        pos = self.gps_to_location(gps)
        (_, curr_command), (near_node, near_command) = self._route_planner.run_step(pos)

        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_curr':curr_command,
                'command_near':near_command,
                'command_near_xy':near_node
                }
        
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['cam_intrinsic'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = self.step
        results['timestamp'] = self.step / 20
        results['box_type_3d'], _ = get_box_type('LiDAR')
  
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['cam_intrinsic'].append(np.matmul(self.lidar2img[cam], np.linalg.inv(self.lidar2cam[cam])))
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        command = tick_data['command_curr']
        results['command'] = command2nohot(tick_data['command_curr'])
        results['ego_fut_cmd'] = command2hot(tick_data['command_curr'])
  
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy
  
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        ego_pose = ego2world
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        results['ego_pose'] = ego_pose
        results['ego_pose_inv'] = ego_pose_inv
        lidar2global = ego2world @ self.lidar2ego
        ego_pose = lidar2global
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        results['ego_pose'] = ego_pose
        results['ego_pose_inv'] = ego_pose_inv
        results['lidar2ego'] = self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        results = self.inference_only_pipeline(results)
        self.device="cuda"
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to(self.device)
            if key == 'input_ids':
                for i in range(len(data[0])):
                    for k in range(len(data[0][i])):
                        # print(data[0][i][k])
                        data[0][i][k] = data[0][i][k].to(self.device)
                    
        custom_wrap_fp16_model(self.model)
        # [StyleDrive] closed-loop has no PKL, so feed the desired driving style here.
        # FORCE_DRIVING_STYLE in [-1,1]: -1 conservative / 0 neutral / +1 aggressive.
        # Nesting matches forward_test's unwrap: data[key][0][0].unsqueeze(0) -> shape (1,).
        # CFG amplification is read separately by the model from the STYLE_GUIDANCE env.
        style_val = self.style_val
        input_data_batch['driving_style'] = [[torch.tensor(style_val, dtype=torch.float32, device=self.device)]]
        output_data_batch = self.model(input_data_batch, return_loss=False)
        out_truck = output_data_batch[0]['pts_bbox']['ego_fut_preds'].cpu().numpy()
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0
        if tick_data['speed']>5:
            throttle_traj = 0
        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan'] = out_truck.tolist()
        self.pid_metadata['command'] = command
        self.pid_metadata['command_near_xy'] = command_near_xy.tolist()
        self.pid_metadata['local_command_xy '] = local_command_xy.tolist()
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info     
        if SAVE_PATH is not None and self.step % 2 == 0:
            self.save(tick_data)
        self.prev_control = control
        # [StyleDrive] live view for VNC: write annotated front cam to /dev/shm.
        # Gated by STYLE_VIEW=1; wrapped so a viz error can never affect driving.
        if os.environ.get('STYLE_VIEW', '0') == '1':
            try:
                self._write_live_view(tick_data, out_truck, control, style_val)
            except Exception:
                pass
        return control

    def _write_live_view(self, tick_data, out_truck, control, style_val):
        # main image = front camera (tick_data imgs are BGR, as in save())
        frame = tick_data['imgs']['CAM_FRONT'].copy()
        H, W = frame.shape[:2]
        lbl = {1: 'AGGRESSIVE', 0: 'neutral', -1: 'conservative'}.get(int(round(style_val)), 'custom')
        g = os.environ.get('STYLE_GUIDANCE', '1.0')
        lines = [
            'STYLE %+.2f (%s)  CFG g=%s' % (style_val, lbl, g),
            'speed %.1f m/s  step %d' % (float(tick_data.get('speed', 0.0)), self.step),
            'thr %.2f  brk %.2f  steer %+.2f' % (control.throttle, control.brake, control.steer),
        ]
        for i, t in enumerate(lines):
            y = 36 + i * 34
            cv2.putText(frame, t, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, t, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        # small top-down trajectory mini-map (forward = up), bottom-right
        try:
            wp = np.asarray(out_truck, dtype=np.float32).reshape(-1, 2)
            ph, pw = 220, 300
            panel = np.zeros((ph, pw, 3), np.uint8)
            scale = 6.0  # px per meter
            ox, oy = pw // 2, ph - 12  # ego at bottom-center
            cv2.circle(panel, (ox, oy), 4, (255, 255, 255), -1)
            prev = (ox, oy)
            for x_fwd, y_left in wp:
                px = int(ox - y_left * scale)   # +y(left) -> screen left
                py = int(oy - x_fwd * scale)    # +x(forward) -> screen up
                px = max(0, min(pw - 1, px)); py = max(0, min(ph - 1, py))
                cv2.line(panel, prev, (px, py), (0, 200, 255), 2)
                cv2.circle(panel, (px, py), 3, (0, 200, 255), -1)
                prev = (px, py)
            cv2.putText(panel, 'plan (3s)', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            frame[H - ph:H, W - pw:W] = panel
        except Exception:
            pass
        out = os.environ.get('ORION_VIEW_FILE', '/dev/shm/orion_view.jpg')
        tmp = out + '.tmp.jpg'
        cv2.imwrite(tmp, frame)            # headless cv2 can encode/write; only GUI is missing
        os.replace(tmp, out)              # atomic -> viewer never reads a half-written file

    def save(self, tick_data):
        # [StyleDrive] called every 2 steps (10 Hz); only the front RGB is written.
        frame = self.step // 2
        cvt_c = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # PIL save uses RGB image
        Image.fromarray(cvt_c(tick_data['imgs']['CAM_FRONT'])).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))

        # metric info (kept: required for the driving-score / ability benchmark)
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])


def command2hot(command,max_dim=6):
    if command < 0:
        command = 4
    command -= 1
    cmd_one_hot = np.zeros(max_dim)
    cmd_one_hot[command] = 1
    return cmd_one_hot

def command2nohot(command,max_dim=6):
    if command < 0:
        command = 4
    command -= 1
    return command

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


custom_fp16 = dict(
                    map_head=False,
                    pts_bbox_head=False)
def custom_wrap_fp16_model(model):
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True
    for module_name, v in custom_fp16.items():
        model._modules[module_name].fp16_enabled = v
