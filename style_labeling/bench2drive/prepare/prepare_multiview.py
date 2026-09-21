"""
Bench2Drive: render a 2x3 surround-view composite (6 cameras) for every frame of the selected scenes.

Output: <clips-dir>/multi_view_images/v1/<scene>/<frame_idx:05d>.jpg

Usage:
    python -m style_labeling.bench2drive.prepare.prepare_multiview
"""
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import os
from collections import defaultdict
from typing import List, Dict, Any
import json

from ...common.paths import B2D_DEFAULT_SCENES_FILE, B2D_ROOT, load_scenes_file

class ClipGenerator:
    """Generate clips from Bench2Drive data for QwenVL analysis."""
    
    def __init__(
        self,
        data_infos: List,
        clip_length: int = 10,          # Number of frames per clip
        sample_interval: int = 5,       # Frame interval (original 10Hz, sample at 2Hz -> interval=5)
        multi_view_dir: str = "styleclips/multi_view_images",  # Directory containing pre-generated multi-view images
        max_distance: float = 50.0,     # Max distance for map/object filtering
    ):
        
        self.data_infos = data_infos
        self.clip_length = clip_length
        self.sample_interval = sample_interval
        self.multi_view_dir = multi_view_dir
        self.max_distance = max_distance
        
        # Group frames by scene
        self.scene_frames = self._group_by_scene()
        
    def _group_by_scene(self) -> Dict[str, List[int]]:
        """Group frame indices by scene/folder."""
        scene_frames = defaultdict(list)
        for idx, info in enumerate(self.data_infos): 
            if(info['folder']=='v1/ParkedObstacle_Town10HD_Route371_Weather7'): # 沒有 HD map，跳過 
                continue
            scene_frames[info['folder']].append(idx)
        return dict(scene_frames) # scene_frames 是把每個 scene 對應的 global frame index 記下來
    
    def get_clips_for_scene(self, scene_name: str) -> List[List[int]]: # 最後會得到某個 scene，每個 clip 的 global index
        
        """
        Get all valid clips for a scene.
        
        Sampling pattern:
        - Clip starting at offset 0: frames [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        - Clip starting at offset 1: frames [1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
        - etc.
        """
        frame_indices = self.scene_frames[scene_name] # scene_frames 是把每個 scene 對應的 global frame index 記下來 
        num_frames = len(frame_indices) # 這個 scene 有幾個 frames
        required_frames = (self.clip_length - 1) * self.sample_interval + 1 # 每個 clip 橫跨的 index 數 (46, ex frame0 ~ frame45)
        
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
    
    def get_frame_info(self, frame_idx: int) -> Dict[str, Any]: # 透過 global frame index 得到 frame info
        
        """Get information for a single frame."""
        sample = self.data_infos[frame_idx]
        
        ego_info = {
            'translation': sample['ego_translation'].tolist() if isinstance(sample['ego_translation'], np.ndarray) else sample['ego_translation'],
            'yaw': ((sample['ego_yaw'])),
            'rotation_rate': ((sample['ego_rotation_rate'])),
            'velocity': ((sample['ego_vel'])),
            'acceleration':((sample['ego_accel'])),
            'command_near': ((sample['command_near'])),
            'command_far': ((sample['command_far'])) if 'command_far' in sample else None,
        }
       
        mask = sample['num_points'] != 0

        objects = []
        for box, name, obj_id in zip(sample['gt_boxes'][mask], sample['gt_names'][mask], sample['gt_ids'][mask]):
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
    
    def get_image_paths_for_clip(self, clip_indices: List[int]) -> List[str]:

        """Get pre-generated multi-view composite image file paths for a clip.
        Images are expected to exist at: {multi_view_dir}/{folder}/{frame_idx:05d}.jpg
        e.g., styleclips/multi_view_images/v1/ConstructionObstacle_Town05_Route68_Weather8/00000.jpg
        """
        paths = []
        for frame_idx in clip_indices:
            sample = self.data_infos[frame_idx]
            folder = sample['folder']  # e.g., 'v1/ConstructionObstacle_Town05_Route68_Weather8'
            local_frame_idx = sample['frame_idx']  # The actual frame index within the scene
            
            # Construct path to pre-existing multi-view image
            multi_view_path = os.path.join(self.multi_view_dir, folder, f"{local_frame_idx:05d}.jpg")
            paths.append(multi_view_path)
        
        return paths
    
    def create_clip_data(self, scene_name: str, clip_indices: List[int]) -> Dict[str, Any]:
        
        """Create complete data dictionary for a clip."""

        local_clip_index = self.data_infos[clip_indices[0]]['frame_idx']
        
        # Get image paths
        video_paths = self.get_image_paths_for_clip(clip_indices) # 這裡的 clip_indices 是指某個 clip 的 global frame indexes
        
        # Get frame-by-frame info
        frames_info = []
        for idx in clip_indices:            
            try:
                frames_info.append(self.get_frame_info(idx))
            except Exception as e:
                print(f"[ERROR] get_frame_info failed at idx={idx}, type={type(e).__name__}, msg={e}")

            
        if not frames_info:
            raise RuntimeError("No valid frame info collected for this clip; check input data and parsing logic.")

        # Calculate statistics
        ego_velocities = [f['ego']['velocity'] for f in frames_info]
        ego_positions = [f['ego']['translation'] for f in frames_info]
        
        if len(ego_positions) >= 2:
            start_pos = np.array(ego_positions[0][:2])
            end_pos = np.array(ego_positions[-1][:2])
            total_displacement = float(np.linalg.norm(end_pos - start_pos))
        else:
            total_displacement = 0.0
        
        return {
            'scene_name': scene_name,
            'clip_frame_indices': clip_indices,
            'local_clip_index': local_clip_index,
            'sample_fps': 2,
            'video_paths': video_paths,
            'frames': frames_info,
            'statistics': {
                'mean_velocity': float(np.mean(ego_velocities)),
                'max_velocity': float(np.max(ego_velocities)),
                'min_velocity': float(np.min(ego_velocities)),
                'velocity_std': float(np.std(ego_velocities)),
                'total_displacement': total_displacement,
                'total_objects_seen': sum(len(f['objects']) for f in frames_info),
                'unique_object_ids': len(set(obj['id'] for f in frames_info for obj in f['objects'])),
            }
        }

    def format_as_qwenvl_message(
        self,
        clip_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        """Format clip data as QwenVL message."""

        local_clip_index = clip_data['local_clip_index']
        video_paths = clip_data['video_paths']
        context_str = ""

        frames_info = clip_data['frames']
        stats = clip_data['statistics']
        ego_velocities = []
        ego_accelerations = []
        ego_rotation_rate = []
        ego_yaws = []
        for frame_info in frames_info:
            ego_velocities.append(frame_info['ego']['velocity'])
            ego_accelerations.append(frame_info['ego']['acceleration'])
            ego_rotation_rate.append(frame_info['ego']['rotation_rate'])
            ego_yaws.append(frame_info['ego']['yaw'])

        context_str = f"""Driving context:
- Ego velocity(x, y, z) from frame 0~9 : {ego_velocities}
- Ego acceleration(x, y, z) from frame 0~9 : {ego_accelerations}
- Ego rotation rate(x, y, z) from frame 0~9 : {ego_rotation_rate}
- Ego yaw from frame 0~9 : {ego_yaws}
- Total displacement: {stats['total_displacement']:.1f} meters
- Objects encountered: {stats['unique_object_ids']} unique objects
"""
        
        return {
            "role": "user",
            "local_clip_index": local_clip_index,
            "content": [
                {
                    "type": "video",
                    "video": video_paths,
                    "sample_fps": str(clip_data['sample_fps']),
                },
                {
                    "type": "text",
                    "text": context_str,
                }
            ],
        }

def CreateMultiview(data_infos, target_scenes, data_root, multi_view_dir):
    for sample in data_infos:
        scene_name = sample['folder'].split('/')[-1] if '/' in sample['folder'] else sample['folder']
        if scene_name not in target_scenes:
            continue

        print(f"Start to create multiview image in {sample['folder']}")
        scene_dir = os.path.join(multi_view_dir, sample['folder'])
        os.makedirs(scene_dir, exist_ok=True)

        cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        output_filename = f"{sample['frame_idx']:05d}.jpg"
        output_path = os.path.join(scene_dir, output_filename)

        # Create the composite image
        fig, axes = plt.subplots(2, 3, figsize=(25, 10))
        axes = axes.flatten()
        for idx, cam_name in enumerate(cameras):
            if cam_name in sample['sensors']:
                cam_info = sample['sensors'][cam_name]
                img_path = os.path.join(data_root, cam_info['data_path'])

                try:
                    img = Image.open(img_path)
                    axes[idx].imshow(img)
                    axes[idx].set_title(cam_name)
                    axes[idx].axis('off')
                except FileNotFoundError:
                    axes[idx].text(0.5, 0.5, f'{cam_name}\nFile not found',
                                    ha='center', va='center')
                    axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, f'{cam_name}\nNot available',
                                ha='center', va='center')
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=100, format='jpg')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--infos-pkl", default=str(B2D_ROOT / "infos" / "b2d_infos_train.pkl"))
    parser.add_argument("--data-root", default=str(B2D_ROOT / "raw"),
                        help="Bench2Drive dataset root; camera 'data_path' entries in the pkl are relative to it.")
    parser.add_argument("--clips-dir", default=str(B2D_ROOT / "StyleClips"))
    parser.add_argument("--scenes-file", default=str(B2D_DEFAULT_SCENES_FILE))
    args = parser.parse_args()

    # Load pkl files
    print("Load pkl file")
    train_infos = pickle.load(open(args.infos_pkl, "rb"))
    print("Succefully load pkls!")

    # Create Multi-view images
    CreateMultiview(train_infos, set(load_scenes_file(args.scenes_file)), args.data_root,
                    os.path.join(args.clips_dir, "multi_view_images"))

if __name__ == "__main__":
    main()
