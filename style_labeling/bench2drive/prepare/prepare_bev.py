"""
Bench2Drive: draw one bird's-eye-view image per frame for every clip of the selected
scenes (map lanes, triggers, ego and object trajectories up to the current frame).

Output: <clips-dir>/BEV_images/<scene>/clip_<start_idx>/<scene>_frame_XX.jpg

Usage:
    python -m style_labeling.bench2drive.prepare.prepare_bev
"""
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches
import hashlib

from ...common.paths import B2D_DEFAULT_SCENES_FILE, B2D_ROOT, load_scenes_file

class ClipGenerator:
    """Generate clips from Bench2Drive data for QwenVL analysis."""
    
    def __init__(
        self,
        data_infos: List,
        map_infos: Dict,
        data_root: str = "",
        clip_length: int = 10,          # Number of frames per clip
        sample_interval: int = 5,       # Frame interval (original 10Hz, sample at 2Hz -> interval=5)
        max_distance: float = 50.0,     # Max distance for map/object filtering
        multi_view_output_dir: str = "StyleClips/multi_view_images",
    ):
        self.data_infos = data_infos
        self.map_infos = map_infos
        self.data_root = data_root
        self.clip_length = clip_length
        self.sample_interval = sample_interval
        self.max_distance = max_distance
        self.multi_view_output_dir = multi_view_output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.multi_view_output_dir, exist_ok=True)
        
        # Group frames by scene
        self.scene_frames = self._group_by_scene()
        
    def _group_by_scene(self) -> Dict[str, List[int]]:
        """Group frame indices by scene/folder."""
        scene_frames = defaultdict(list)
        for idx, info in enumerate(self.data_infos):
            scene_frames[info['folder']].append(idx)
        # print(f"scene frames: {scene_frames}")
        return dict(scene_frames)
    
    def get_clips_for_scene(self, scene_name: str) -> List[List[int]]:
        """
        Get all valid clips for a scene.
        
        Sampling pattern:
        - Clip starting at offset 0: frames [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        - Clip starting at offset 1: frames [1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
        - etc.
        """
        frame_indices = self.scene_frames[scene_name] 
        num_frames = len(frame_indices) 
        required_frames = (self.clip_length - 1) * self.sample_interval + 1 
        
        clips = []
        
        # Only iterate over valid starting offsets
        for offset in range(num_frames - required_frames + 1):
            start = offset
            # Create clip with sampled frames
            clip_local_indices = [
                start + i * self.sample_interval 
                for i in range(self.clip_length)
            ]
            # Convert to global indices
            clip_global_indices = [frame_indices[i] for i in clip_local_indices]
            clips.append(clip_global_indices)
        
        return clips # list of list (第二個 list 指的是該 clip 每個 frame 的 global index)
    
    def get_all_clips(self) -> List[Tuple[str, List[int]]]:
        """Get all valid clips from all scenes."""
        all_clips = []
        for scene_name in self.scene_frames.keys():
            clips = self.get_clips_for_scene(scene_name)
            for clip in clips:
                all_clips.append((scene_name, clip))
        return all_clips
    
    def get_frame_info(self, frame_idx: int) -> Dict[str, Any]:
        """Get information for a single frame."""
        sample = self.data_infos[frame_idx]
        # print(sample)
        # Basic ego info
        print(sample['ego_translation'].tolist())
        ego_info = {
            'translation': sample['ego_translation'].tolist() if isinstance(sample['ego_translation'], np.ndarray) else sample['ego_translation'],
            'yaw': ((sample['ego_yaw'])),
            'velocity': ((sample['ego_vel'])),
            'command_near': ((sample['command_near'])),
            'command_far': ((sample['command_far'])) if 'command_far' in sample else None,
        }
       
        # # Object info (filter by valid objects)
        mask = sample['num_points'] != 0
        # print(f"in get frame ingo: mask: {mask}")
        objects = []
        for i, (box, name, obj_id) in enumerate(zip(
            sample['gt_boxes'][mask],
            sample['gt_names'][mask],
            sample['gt_ids'][mask]
        )):
            distance = np.sqrt(box[0]**2 + box[1]**2)
            if distance <= self.max_distance:
                objects.append({
                    'id': int((obj_id)),
                    'class': str(name),
                    'position': [(box[0]), (box[1]), (box[2])],
                    'size': [(box[3]), (box[4]), (box[5])],
                    'yaw': (box[6]),
                    'velocity': [(box[7]), (box[8])],
                    'distance': (distance),
                })
        
        
        return {
            'frame_idx': sample['frame_idx'],
            'ego': ego_info,
            'objects': objects,
        }
    
    def get_trajectory_for_a_clip(self, clip_indices: List[int], train_infos) -> Dict :
        past_frames = 2
        future_frames = 7
        total_frames = len(clip_indices)
        cur_idx = clip_indices[2]
        cur_frame = train_infos[cur_idx]

        # Transform matrix from world -> lidar (of current frame)
        world2lidar_cur = np.array(cur_frame['sensors']['LIDAR_TOP']['world2lidar'])
            

        # ========== 1. Collect Ego Trajectory ==========
        ego_positions = np.zeros((total_frames, 2))
        ego_masks = np.zeros(total_frames)
        ego_velocities = np.zeros(total_frames)
        
        for frame_idx, adj_idx in enumerate(clip_indices):
            
            adj_frame = train_infos[adj_idx]
            
            # Get ego position in world coordinate
            ego_pos_world = np.array(adj_frame['ego_translation'])
            
            # Transform to lidar coordinate of current frame
            ego_pos_world_homo = np.concatenate([ego_pos_world, [1.0]])
            ego_pos_lidar = (world2lidar_cur @ ego_pos_world_homo)[:2]
            
            ego_positions[frame_idx] = ego_pos_lidar
            ego_masks[frame_idx] = 1
            ego_velocities[frame_idx] = adj_frame['ego_vel']
            
        # Calculate offset (movement between frames)
        offset_track = ego_positions[1:] - ego_positions[:-1]

        # Accumulate offset for past trajectory (from far to near)      # Numbers from small to large represent points from far to near
        ego_his_trajs = np.zeros((past_frames, 2))
        for j in range(past_frames-1, -1, -1):                          # From 1 ~ 0 (actually just two steps)
            if j == past_frames - 1:                                    # When j = 1
                ego_his_trajs[j] = -offset_track[j]                     # Displacement from current frame to previous frame # 要加負號才對
            else:                                                       # When j = 0
                ego_his_trajs[j] = -(ego_his_trajs[j+1] + offset_track[j]) # Displacement from current frame to two frames ago (accumulated)
        
        # Accumulate offset for future trajectory (from near to far)    # Numbers from small to large represent points from near to far
        ego_fut_trajs = np.zeros((future_frames, 2))
        for j in range(past_frames, past_frames + future_frames):       # From 2 ~ 8
            if j == past_frames:                                        
                ego_fut_trajs[j - past_frames] = offset_track[j]
            else:
                ego_fut_trajs[j - past_frames] = ego_fut_trajs[j - past_frames - 1] + offset_track[j]
        
        ego_fut_masks = ego_masks[past_frames+1:]                   # Not sure what mask is for yet
        
        
        # ========== 2. Collect NPC Trajectory ==========
        # Collect all object IDs across all frames
        object_data = {}  # {object_id: {'positions': [], 'masks': [], 'boxes': [], 'name': str}}

        for frame_idx, adj_idx in enumerate(clip_indices):
        
            adj_frame = train_infos[adj_idx]
            
            # Get transformation from adjacent frame's lidar to current frame's lidar
            world2lidar_adj = np.array(adj_frame['sensors']['LIDAR_TOP']['world2lidar'])
            lidar2world_adj = np.linalg.inv(world2lidar_adj)
            
            # For each object in this frame
            gt_boxes = adj_frame['gt_boxes']
            gt_names = adj_frame['gt_names']
            gt_ids = adj_frame['gt_ids']
            num_points = adj_frame['num_points']
            
            for i in range(len(gt_boxes)):
                obj_id = gt_ids[i]
                obj_name = gt_names[i]
                obj_box = gt_boxes[i]  # [x, y, z, w, l, h, yaw, vx, vy]
                has_points = num_points[i] > 0
                
                # Initialize object entry if not exists
                if obj_id not in object_data:
                    object_data[obj_id] = {
                        'positions': np.zeros((total_frames, 2)),
                        'masks': np.zeros(total_frames),
                        'boxes': np.zeros((total_frames, 9)),
                        'name': obj_name
                    }
                
                # Position in adjacent frame's lidar coordinate
                pos_adj_lidar = np.array([obj_box[0], obj_box[1], obj_box[2], 1.0])
                
                # Transform to world coordinate
                pos_world = lidar2world_adj @ pos_adj_lidar
                
                # Transform to current frame's lidar coordinate
                pos_cur_lidar = world2lidar_cur @ pos_world
                
                object_data[obj_id]['positions'][frame_idx] = pos_cur_lidar[:2]
                object_data[obj_id]['masks'][frame_idx] = 1 if has_points else 0.5  # 0.5 for no points
                object_data[obj_id]['boxes'][frame_idx] = obj_box

    
    def get_map_info_for_frame(self, frame_idx: int) -> Dict[str, Any]:
        """Get map information around the ego vehicle for a single frame."""
        sample = self.data_infos[frame_idx]
        town_name = sample['town_name']
        map_info = self.map_infos[town_name]
        
        world2lidar = np.array(sample['sensors']['LIDAR_TOP']['world2lidar'])
        ego_xy = np.linalg.inv(world2lidar)[0:2, 3]
        
        # Count nearby lanes and triggers
        nearby_lane_types = []
        for i in range(len(map_info['lane_sample_points'])):
            sample_points = map_info['lane_sample_points'][i]
            distance = np.linalg.norm(sample_points[:, 0:2] - ego_xy, axis=-1)
            if distance.min() < self.max_distance:
                nearby_lane_types.append(map_info['lane_types'][i])
        
        nearby_trigger_types = []
        for i in range(len(map_info['trigger_volumes_sample_points'])):
            sample_points = map_info['trigger_volumes_sample_points'][i]
            distance = np.linalg.norm(sample_points[0:2] - ego_xy, axis=-1)
            if distance.min() < self.max_distance:
                nearby_trigger_types.append(map_info['trigger_volumes_types'][i])
        
        return {
            'town_name': town_name,
            'num_lanes': len(nearby_lane_types),
            'num_triggers': len(nearby_trigger_types),
            'lane_types': list(set(nearby_lane_types)),
            'trigger_types': list(set(nearby_trigger_types)),
        }

def create_per_frame_bev(
    data_infos: List,
    map_infos: Dict,
    clip_indices: List[int],
    output_dir: str,
    scene_name: str,
    max_distance: float = 50.0,
    figsize: Tuple[int, int] = (14, 14),
    ego_width: float = 2.0,
    ego_length: float = 4.5,
    show_map: bool = True,
):
    
    os.makedirs(output_dir, exist_ok=True)
    
    ref_idx = clip_indices[0]
    ref_frame = data_infos[ref_idx]
    world2lidar_ref = np.array(ref_frame['sensors']['LIDAR_TOP']['world2lidar'])
    
    ego_positions = []
    ego_yaws = []
    for frame_idx in clip_indices:
        frame = data_infos[frame_idx]
        ego_pos_world = np.array(frame['ego_translation']).copy()
        ego_pos_world[2] = ref_frame['ego_translation'][2] # Flatten Z
        ego_pos_world_homo = np.concatenate([ego_pos_world, [1.0]])
        ego_pos_lidar = (world2lidar_ref @ ego_pos_world_homo)[:2]
        ego_positions.append(ego_pos_lidar)
        
        v_world = np.array([np.cos(frame['ego_yaw']), np.sin(frame['ego_yaw'])])
        v_ref_lidar = world2lidar_ref[:2, :2] @ v_world
        ego_yaws.append(np.arctan2(v_ref_lidar[1], v_ref_lidar[0]))
        
    ego_positions = np.array(ego_positions)
    
    object_trajectories = {}
    for clip_frame_num, frame_idx in enumerate(clip_indices):
        frame = data_infos[frame_idx]
        world2lidar_frame = np.array(frame['sensors']['LIDAR_TOP']['world2lidar'])
        lidar2world_frame = np.linalg.inv(world2lidar_frame)
        
        gt_boxes = frame['gt_boxes']
        gt_names = frame['gt_names']
        gt_ids = frame['gt_ids']
        num_points = frame['num_points']
        
        for i in range(len(gt_boxes)):
            obj_id = gt_ids[i]
            obj_name = gt_names[i]
            obj_box = gt_boxes[i]
            
            is_static_or_traffic = obj_name.startswith('static.') or obj_name.startswith('traffic.')
            has_points = num_points[i] > 0
            
            if not (has_points or is_static_or_traffic):
                continue
                
            pos_frame_lidar = np.array([obj_box[0], obj_box[1], obj_box[2], 1.0])
            pos_world = lidar2world_frame @ pos_frame_lidar
            ##### pos_world[2] = ref_frame['ego_translation'][2] # Flatten Z
            pos_ref_lidar = world2lidar_ref @ pos_world
            
            if np.linalg.norm(pos_ref_lidar[:2]) > max_distance * 2:
                continue
            
            v_local_lidar = np.array([np.sin(obj_box[6]), np.cos(obj_box[6])])
            trans_matrix = world2lidar_ref @ lidar2world_frame
            v_ref_lidar = trans_matrix[:2, :2] @ v_local_lidar
            yaw_transformed = np.arctan2(v_ref_lidar[1], v_ref_lidar[0])
            
            if obj_id not in object_trajectories:
                object_trajectories[obj_id] = {
                    'positions': [], 'yaws': [], 'widths': [], 'lengths': [],
                    'name': obj_name, 'clip_frame_nums': []
                }
            
            object_trajectories[obj_id]['positions'].append(pos_ref_lidar[:2])
            object_trajectories[obj_id]['yaws'].append(yaw_transformed)
            object_trajectories[obj_id]['widths'].append(obj_box[3])
            object_trajectories[obj_id]['lengths'].append(obj_box[4])
            object_trajectories[obj_id]['clip_frame_nums'].append(clip_frame_num)
            
    nearby_lanes = []
    nearby_lane_types = []
    nearby_triggers = []
    nearby_trigger_types = []
    
    if show_map:
        town_name = ref_frame['town_name']
        map_info = map_infos[town_name]
        ego_xy = np.linalg.inv(world2lidar_ref)[0:2, 3]
        
        for idx in range(len(map_info['lane_sample_points'])):
            sample_points = map_info['lane_sample_points'][idx]
            if np.linalg.norm(sample_points[:, 0:2] - ego_xy, axis=-1).min() < max_distance:
                lane_points_world = map_info['lane_points'][idx].copy()
                lane_points_world[:, 2] = ref_frame['ego_translation'][2]
                lane_points_homo = np.concatenate([lane_points_world, np.ones((len(lane_points_world), 1))], axis=-1)
                nearby_lanes.append((world2lidar_ref @ lane_points_homo.T).T[:, :2])
                nearby_lane_types.append(map_info['lane_types'][idx])
                
        for idx in range(len(map_info['trigger_volumes_sample_points'])):
            sample_points = map_info['trigger_volumes_sample_points'][idx]
            if len(sample_points) == 0:
                continue
            if np.linalg.norm(sample_points[0:2] - ego_xy, axis=-1).min() < max_distance:
                trigger_points_world = map_info['trigger_volumes_points'][idx].copy()
                trigger_points_world[:, 2] = ref_frame['ego_translation'][2]
                trigger_points_homo = np.concatenate([trigger_points_world, np.ones((len(trigger_points_world), 1))], axis=-1)
                nearby_triggers.append((world2lidar_ref @ trigger_points_homo.T).T[:, :2])
                nearby_trigger_types.append(map_info['trigger_volumes_types'][idx])
                
    def get_color_for_class(class_name):
        class_name_lower = class_name.lower()
        if 'pedestrian' in class_name_lower or 'walker' in class_name_lower: return "#DAE93E"
        elif 'bicycle' in class_name_lower or 'crossbike' in class_name_lower or 'diamondback' in class_name_lower: return "#741BBE"
        elif 'motorcycle' in class_name_lower or 'harley' in class_name_lower or 'kawasaki' in class_name_lower or 'vespa' in class_name_lower or 'yamaha' in class_name_lower: return "#E01D1D"
        elif 'bus' in class_name_lower or 'fusorosa' in class_name_lower: return "#4BC3EF"
        elif 'truck' in class_name_lower or 'carlamotors' in class_name_lower or 'cybertruck' in class_name_lower: return "#A38209"
        elif 'van' in class_name_lower or 'ambulance' in class_name_lower or 'sprinter' in class_name_lower or 'volkswagen.t2' in class_name_lower: return "#26A41A"
        elif class_name_lower.startswith('static.prop'): return "#FFD700" # Yellow for static props like construction warning
        elif class_name_lower.startswith('traffic.'): return "#FF0000" # Red for traffic elements like speed limits
        else: return "#0F29E9"

    def str_to_color(s):
        hash_val = hashlib.md5(s.encode('utf-8')).hexdigest()
        return '#' + hash_val[:6]

    output_paths = []
    
    for current_i in range(len(clip_indices)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        if show_map:
            lane_color_map = {'Broken': '#888888', 'Solid': '#000000', 'SolidSolid': '#CC8800', 'Center': '#FF5500', 'NONE': '#000000'}
            lane_linestyle_map = {'Broken': '--', 'Solid': '-', 'SolidSolid': '-', 'Center': '-', 'NONE': '-'}
            
            for lane_pts, lane_type in zip(nearby_lanes, nearby_lane_types):
                color = lane_color_map.get(lane_type, '#888888')
                linestyle = lane_linestyle_map.get(lane_type, '-')
                ax.plot(lane_pts[:, 0], lane_pts[:, 1], color=color, linewidth=1.2, alpha=0.6, linestyle=linestyle, zorder=1)
                
            for trig_pts, trig_type in zip(nearby_triggers, nearby_trigger_types):
                if trig_type == 'TrafficLight':
                    color = '#008800'
                elif trig_type == 'StopSign':
                    color = '#CC0000'
                else:
                    color = str_to_color(trig_type)
                ax.fill(trig_pts[:, 0], trig_pts[:, 1], color=color, alpha=0.25, zorder=2)
                ax.plot(np.append(trig_pts[:, 0], trig_pts[0, 0]), np.append(trig_pts[:, 1], trig_pts[0, 1]), color=color, linewidth=1.0, alpha=0.6, zorder=2)

        if current_i > 0:
            ax.plot(ego_positions[:current_i+1, 0], ego_positions[:current_i+1, 1], 
                    color='green', linewidth=4, linestyle='--', alpha=0.8, zorder=5)
            
        ego_rect = Rectangle((-ego_length/2, -ego_width/2), ego_length, ego_width,
                             linewidth=2, edgecolor='green', facecolor='green', alpha=0.5)
        t = Affine2D().rotate(ego_yaws[current_i]).translate(ego_positions[current_i, 0], ego_positions[current_i, 1]) + ax.transData
        ego_rect.set_transform(t)
        ax.add_patch(ego_rect)
        
        ax.scatter(ego_positions[current_i, 0], ego_positions[current_i, 1], 
                   color='green', s=100, marker='*', edgecolors='white', 
                   linewidths=2, zorder=11)

        for obj_id, obj_data in object_trajectories.items():
            positions = np.array(obj_data['positions'])
            clip_frame_nums = obj_data['clip_frame_nums']
            
            if current_i not in clip_frame_nums:
                continue
            
            valid_indices = [idx for idx, f_num in enumerate(clip_frame_nums) if f_num <= current_i]
            if not valid_indices:
                continue
                
            cur_obj_positions = positions[valid_indices]
            color = get_color_for_class(obj_data['name'])
            
            if len(cur_obj_positions) >= 2:
                ax.plot(cur_obj_positions[:, 0], cur_obj_positions[:, 1], 
                        color=color, linewidth=3.5, linestyle='--', alpha=0.9, zorder=5)
            
            idx_at_current = clip_frame_nums.index(current_i)
            w = obj_data['widths'][idx_at_current]
            l = obj_data['lengths'][idx_at_current]
            yaw = obj_data['yaws'][idx_at_current]
            pos = positions[idx_at_current]
            
            rect = Rectangle((-l/2, -w/2), l, w,
                             linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.4)
            t = Affine2D().rotate(yaw).translate(pos[0], pos[1]) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
            
            ax.scatter(pos[0], pos[1], color=color, s=60, marker='o', edgecolors='white', linewidths=1, alpha=0.9, zorder=9)

        ax.set_xlim(-max_distance, max_distance)
        ax.set_ylim(-max_distance, max_distance)
        ax.set_xlabel('X (m)', fontsize=12, color='black')
        ax.set_ylabel('Y (m)', fontsize=12, color='black')
        ax.set_title(f'Per Frame BEV - Frame {current_i}', fontsize=14, color='black')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='gray')

        # === Legend Setup ===
        legend_elements = []
        if show_map and len(nearby_lanes) > 0:
            legend_elements.append(plt.Line2D([0], [0], color='white', linewidth=0, label='── Map ──'))
            lane_types_shown = set(nearby_lane_types)
            if 'Broken' in lane_types_shown: legend_elements.append(plt.Line2D([0], [0], color='#888888', linewidth=2, linestyle='--', label='Broken Lane'))
            if 'Solid' in lane_types_shown: legend_elements.append(plt.Line2D([0], [0], color='#000000', linewidth=2, linestyle='-', label='Solid Lane'))
            if 'SolidSolid' in lane_types_shown: legend_elements.append(plt.Line2D([0], [0], color='#CC8800', linewidth=2, linestyle='-', label='Double Solid'))
            if 'Center' in lane_types_shown: legend_elements.append(plt.Line2D([0], [0], color='#FF5500', linewidth=2, linestyle='-', label='Center Line'))
        
        if show_map and len(nearby_triggers) > 0:
            trigger_types_shown = sorted(set(nearby_trigger_types))
            for trig_type in trigger_types_shown:
                if trig_type == 'TrafficLight':
                    color = '#008800'
                elif trig_type == 'StopSign':
                    color = '#CC0000'
                else:
                    color = str_to_color(trig_type)
                legend_elements.append(mpatches.Patch(facecolor=color, alpha=0.25, edgecolor=color, label=trig_type))
        
        if len(legend_elements) > 0:
            legend_elements.append(plt.Line2D([0], [0], color='white', linewidth=0, label='── Vehicles & Objects ──'))
        
        legend_elements.extend([
            mpatches.Patch(facecolor='green', alpha=0.5, label='Ego Vehicle'),
            mpatches.Patch(facecolor="#0F29E9", alpha=0.4, label='Car'),
            mpatches.Patch(facecolor="#26A41A", alpha=0.4, label='Van'),
            mpatches.Patch(facecolor="#A38209", alpha=0.4, label='Truck'),
            mpatches.Patch(facecolor="#4BC3EF", alpha=0.4, label='Bus'),
            mpatches.Patch(facecolor="#E01D1D", alpha=0.4, label='Motorcycle'),
            mpatches.Patch(facecolor="#741BBE", alpha=0.4, label='Bicycle'),
            mpatches.Patch(facecolor="#DAE93E", alpha=0.4, label='Pedestrian'),
            mpatches.Patch(facecolor="#FFD700", alpha=0.4, label='Static Prop (Obstacle)'),
            mpatches.Patch(facecolor="#FF0000", alpha=0.4, label='Traffic Sign (Speed Limit)'),
        ])
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                  facecolor='white', labelcolor='black', framealpha=0.9, edgecolor='black')
        # ====================
        
        # Save
        scene_name_safe = scene_name.replace("/", "_")
        output_path = os.path.join(output_dir, f"{scene_name_safe}_frame_{current_i:02d}.jpg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, format='jpg', bbox_inches='tight')
        plt.close(fig)
        
        output_paths.append(output_path)
    
    return output_paths

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--infos-pkl", default=str(B2D_ROOT / "infos" / "b2d_infos_train.pkl"))
    parser.add_argument("--map-pkl", default=str(B2D_ROOT / "infos" / "b2d_map_infos.pkl"))
    parser.add_argument("--clips-dir", default=str(B2D_ROOT / "StyleClips"))
    parser.add_argument("--scenes-file", default=str(B2D_DEFAULT_SCENES_FILE))
    args = parser.parse_args()

    output_base_path = os.path.join(args.clips_dir, "BEV_images")

    # Load pkl files
    train_infos = pickle.load(open(args.infos_pkl, "rb"))
    map_infos = pickle.load(open(args.map_pkl, "rb"))
    print("Succefully load pkls!")

    generator = ClipGenerator(
        data_infos = train_infos,
        map_infos = map_infos,
        data_root = "bench2drive/",
        clip_length = 10,       # 10 frames per clip (5 seconds at 2Hz)
        sample_interval = 5,    # Sample every 5 frames (2Hz from 10Hz original)
        multi_view_output_dir = os.path.join(args.clips_dir, "multi_view_images"),  # the class mkdirs this; keep it inside clips-dir
    )
    
    total_clip = 0
    for target_scene in load_scenes_file(args.scenes_file):
        target_scene_key = f"v1/{target_scene}"
        if target_scene_key not in generator.scene_frames:
            print(f"❌❌❌Warning: Scene '{target_scene}' not found in the data. Skipping.")
            continue
        
        scene_clips = generator.get_clips_for_scene(target_scene_key)
        scene_clip_num = len(scene_clips)
        print(f"Creating per-frame BEVs for scene: {target_scene}")

        for target_clip_idx in range(0, scene_clip_num, 1):
            target_clip_indices = scene_clips[target_clip_idx]

            target_scene_safe = target_scene.replace("/", "_").replace("v1_", "")
            output_dir = os.path.join(output_base_path, f"{target_scene_safe}", f"clip_{target_clip_idx}")

            generated_files = create_per_frame_bev(
                data_infos = generator.data_infos,
                map_infos = map_infos,
                clip_indices = target_clip_indices,
                output_dir = output_dir,
                scene_name = target_scene,
                max_distance = 50.0,
                show_map = True,
            )            

        target_scene_safe = target_scene.replace("/", "_").replace("v1_", "")
        print(f"{target_scene_safe}, clip#: {scene_clip_num}")
        total_clip += scene_clip_num
    
    print(f"total clip#: {total_clip}")

if __name__ == "__main__":
    main()
