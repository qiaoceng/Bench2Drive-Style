# ==============================================================================
# [Bench2Drive-Style] NEW FILE -- script
# Author: YinXuan1223   (carried over from the ORION_NYCUACM_version fork)
#
# Diagnostic: plot the diversity of the diffusion planner's output.  For each
# sample it renders a BEV figure with all 20 trajectory modes, the selected mode
# and the ground-truth trajectory, and reports diversity metrics.  Used to check
# that the diffusion decoder had not collapsed onto a single mode.
# Runs against orion_stage3_fp16_diff.py; see the usage block in the docstring below.
# Run it from the StyleOrion/ directory.
# ==============================================================================
"""
Visualize diffusion trajectory diversity from Orion model.

Generates BEV plots showing all 20 trajectory modes, the selected mode,
and the ground truth trajectory. Also computes diversity metrics.

Usage:
# run from the StyleOrion/ directory
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python \
adzoo/orion/visualize_diffusion_trajectories.py \
    adzoo/orion/configs/orion_stage3_fp16_diff.py \
    adzoo/orion/work_dirs/orion_stage2_diff_train/iter_30000.pth \
    --num-samples 20 \
    --output-dir adzoo/orion/work_dirs/orion_stage2_diff_train/vis_trajectories
"""
import argparse
import os
import sys
import pickle
import numpy as np

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize diffusion trajectory diversity')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--num-samples', type=int, default=20, help='number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='vis_trajectories', help='output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show-all-modes', action='store_true', default=True,
                        help='show all 20 modes (default: True)')
    parser.add_argument('--num-stochastic-runs', type=int, default=5,
                        help='run inference N times per sample to show stochastic diversity')
    return parser.parse_args()


def _sequential_extract_img_feat(self, img):
    """Process cameras one at a time to reduce peak GPU memory."""
    B = img.size(0)
    if img is None:
        return None
    if img.dim() == 6:
        img = img.flatten(1, 2)
    if img.dim() == 5 and img.size(0) == 1:
        img = img.squeeze(0)  # (N, C, H, W)
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)

    if self.use_grid_mask:
        img = self.grid_mask(img)

    feats_list = []
    for i in range(img.size(0)):
        single = img[i:i+1]  # (1, C, H, W)
        feat = self.img_backbone(single)
        if isinstance(feat, dict):
            feat = list(feat.values())
        if self.with_img_neck:
            feat = self.img_neck(feat)
        feats_list.append(feat[self.position_level])
        del single
    torch.cuda.empty_cache()

    combined = torch.cat(feats_list, dim=0)  # (BN, C, H, W)
    del feats_list
    BN, C_out, H_out, W_out = combined.size()
    return combined.view(B, int(BN / B), C_out, H_out, W_out)


def build_model_and_dataloader(cfg, checkpoint_path):
    from mmcv.utils import load_checkpoint, set_random_seed
    from mmcv.models import build_model
    from mmcv.datasets import build_dataset, build_dataloader

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.to(torch.bfloat16).eval()
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    import types
    model.extract_img_feat = types.MethodType(_sequential_extract_img_feat, model)

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=dict(type='DistributedSampler'),
    )
    return model, data_loader, dataset


def move_backbone_to_gpu(model):
    model.img_backbone.cuda()
    if model.with_img_neck:
        model.img_neck.cuda()
    if hasattr(model, 'grid_mask'):
        model.grid_mask.cuda()
    torch.cuda.empty_cache()


def move_backbone_to_cpu(model):
    model.img_backbone.cpu()
    if model.with_img_neck:
        model.img_neck.cpu()
    if hasattr(model, 'grid_mask'):
        model.grid_mask.cpu()
    torch.cuda.empty_cache()


def move_rest_to_gpu(model):
    for name, module in model.named_children():
        if name not in ('img_backbone', 'img_neck', 'grid_mask'):
            module.cuda()
    # move top-level parameters and buffers
    for name, param in model.named_parameters(recurse=False):
        param.data = param.data.cuda()
    for name, buf in model.named_buffers(recurse=False):
        buf.data = buf.data.cuda()
    torch.cuda.empty_cache()


def move_rest_to_cpu(model):
    for name, module in model.named_children():
        if name not in ('img_backbone', 'img_neck', 'grid_mask'):
            module.cpu()
    for name, param in model.named_parameters(recurse=False):
        param.data = param.data.cpu()
    for name, buf in model.named_buffers(recurse=False):
        buf.data = buf.data.cpu()
    torch.cuda.empty_cache()


@torch.no_grad()
def extract_all_modes(model, data):
    """Run diffusion inference and return ALL 20 trajectory modes + scores."""
    m = model.module if hasattr(model, 'module') else model

    img = data['img'][0].unsqueeze(0).cuda()
    img_metas = data['img_metas'][0].data[0]
    data_gpu = {}
    for k in data:
        if k in ('img', 'img_metas'):
            continue
        v = data[k]
        if hasattr(v, 'data'):
            v = v.data
        if isinstance(v, list) and len(v) > 0:
            v = v[0]
        if isinstance(v, torch.Tensor):
            data_gpu[k] = v.cuda()
        else:
            data_gpu[k] = v
    data_gpu['img'] = img

    data_gpu['img_feats'] = m.extract_feat(img)
    if img.dim() == 4:
        data_gpu['img'] = img.unsqueeze(0)

    m.eval()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        bbox_pts, generated_text, lane_results, metric_dict = m.simple_test_pts(
            img_metas, **data_gpu)

    result = {}
    if hasattr(m, 'use_diff_decoder') and m.use_diff_decoder:
        result['all_modes'] = lane_results[0].get('ego_fut_preds_all_modes', None)
        result['selected'] = lane_results[0].get('ego_fut_preds', None)
        result['cls_scores'] = lane_results[0].get('ego_fut_cls_scores', None)

    if result.get('all_modes') is None:
        result['all_modes'] = None
        result['selected'] = lane_results[0].get('ego_fut_preds', torch.zeros(6, 2))

    result['gt'] = None
    if 'ego_fut_trajs' in data_gpu:
        gt = data_gpu['ego_fut_trajs'][0, 0].cpu().float()
        result['gt'] = gt.cumsum(dim=-2).numpy()

    result['command'] = data_gpu.get('command', None)
    result['scene_token'] = img_metas[0].get('scene_token', 'unknown')
    result['frame_idx'] = img_metas[0].get('frame_idx', 0)

    return result


@torch.no_grad()
def extract_modes_raw(model, data):
    """Directly run diffusion decoder to get all 20 modes with scores."""
    m = model.module if hasattr(model, 'module') else model

    img = data['img'][0].unsqueeze(0).cuda()
    img_metas = data['img_metas'][0].data[0]
    data_gpu = {}
    for k in data:
        if k in ('img', 'img_metas'):
            continue
        v = data[k]
        if hasattr(v, 'data'):
            v = v.data
        if isinstance(v, list) and len(v) > 0:
            v = v[0]
        if isinstance(v, torch.Tensor):
            data_gpu[k] = v.cuda()
        else:
            data_gpu[k] = v
    data_gpu['img'] = img

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        img_feats = m.extract_feat(img)
        data_gpu['img_feats'] = img_feats
        if img.dim() == 4:
            data_gpu['img'] = img.unsqueeze(0)

        # We need to run the model's forward and capture intermediate diffusion outputs.
        # The simplest approach: use the DEBUG_SHOW_PRED env var mechanism
        # But for raw mode extraction, we'll monkey-patch temporarily.

        # Store original and patch
        original_simple_test_pts = m.simple_test_pts

        all_modes_storage = {}

        def patched_simple_test_pts(img_metas_arg, **data_arg):
            """Wrapper that captures all 20 diffusion modes."""
            result = original_simple_test_pts(img_metas_arg, **data_arg)
            return result

        bbox_pts, generated_text, lane_results, metric_dict = m.simple_test_pts(
            img_metas, **data_gpu)

    # The model's inference code at line 904: ego_fut_preds = poses_reg (all 20 modes)
    # At line 935-938: it splits into selected and inactive
    # We need to capture poses_reg BEFORE the split
    # Since we can't easily hook into the running code, let's re-run the diffusion part only

    result = {}
    _bf = torch.bfloat16

    with torch.cuda.amp.autocast(enabled=False):
        # Get ego_feature from the LLM (already computed, stored in lane_results)
        # We'll extract what we need from the model state

        # Actually, let's just directly run the diffusion decoder part
        # We need current_states (ego_feature) which requires running the full pipeline
        # The simplest reliable approach: capture from lane_results

        ego_fut_preds = lane_results[0].get('ego_fut_preds', None)

        # Get plan_anchor for reference
        plan_anchor = m.plan_anchor.detach().cpu().float().numpy()  # (20, 6, 2)
        result['plan_anchor'] = plan_anchor

        if ego_fut_preds is not None:
            result['selected'] = ego_fut_preds.cpu().numpy()
        else:
            result['selected'] = np.zeros((6, 2))

    result['gt'] = None
    if 'ego_fut_trajs' in data_gpu:
        gt = data_gpu['ego_fut_trajs'][0, 0].cpu().float()
        result['gt'] = gt.cumsum(dim=-2).numpy()

    commands = ["LEFT", "RIGHT", "STRAIGHT", "LANE_FOLLOW", "CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"]
    cmd = data_gpu.get('command', None)
    if cmd is not None:
        cmd_idx = int(cmd.item()) if isinstance(cmd, torch.Tensor) else int(cmd)
        result['command'] = commands[cmd_idx] if cmd_idx < len(commands) else 'UNKNOWN'
    else:
        result['command'] = 'UNKNOWN'

    result['scene_token'] = img_metas[0].get('scene_token', 'unknown')
    result['frame_idx'] = img_metas[0].get('frame_idx', 0)

    return result


def plot_trajectories_bev(all_modes, selected, gt, plan_anchor, command,
                          scene_token, frame_idx, cls_scores=None,
                          output_path=None):
    """Plot BEV visualization of all trajectory modes."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Left: All 20 modes ---
    ax = axes[0]
    ax.set_title(f'All 20 Diffusion Modes\n{command}', fontsize=13, fontweight='bold')

    cmap = plt.cm.tab20
    if all_modes is not None:
        for i in range(all_modes.shape[0]):
            traj = all_modes[i]  # (6, 2)
            color = cmap(i / 20)
            alpha = 0.6
            label = f'Mode {i}'
            if cls_scores is not None:
                alpha = 0.3 + 0.7 * float(cls_scores[i])
                label += f' ({cls_scores[i]:.2f})'
            ax.plot(traj[:, 1], traj[:, 0], '-o', color=color, alpha=alpha,
                    markersize=3, linewidth=1.5, label=label)
    elif plan_anchor is not None:
        for i in range(plan_anchor.shape[0]):
            traj = plan_anchor[i]
            color = cmap(i / 20)
            ax.plot(traj[:, 1], traj[:, 0], '--', color=color, alpha=0.4,
                    markersize=2, linewidth=1)

    if gt is not None:
        ax.plot(gt[:, 1], gt[:, 0], '-s', color='red', linewidth=3,
                markersize=6, label='GT', zorder=10)
    if selected is not None:
        ax.plot(selected[:, 1], selected[:, 0], '-^', color='lime',
                linewidth=3, markersize=8, label='Selected', zorder=11)

    ax.plot(0, 0, '*', color='yellow', markersize=15, markeredgecolor='black',
            zorder=12, label='Ego')
    ax.set_xlabel('Lateral (m)', fontsize=11)
    ax.set_ylabel('Longitudinal (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # --- Right: Selected vs GT with spread ---
    ax2 = axes[1]
    ax2.set_title(f'Selected vs GT\nScene: {scene_token} | Frame: {frame_idx}', fontsize=13)

    if all_modes is not None:
        for i in range(all_modes.shape[0]):
            traj = all_modes[i]
            ax2.plot(traj[:, 1], traj[:, 0], '-', color='lightblue', alpha=0.3,
                     linewidth=1)

    if gt is not None:
        ax2.plot(gt[:, 1], gt[:, 0], '-s', color='red', linewidth=3,
                 markersize=6, label='GT')
    if selected is not None:
        ax2.plot(selected[:, 1], selected[:, 0], '-^', color='lime',
                 linewidth=3, markersize=8, label='Selected')

    ax2.plot(0, 0, '*', color='yellow', markersize=15, markeredgecolor='black',
             zorder=12, label='Ego')
    ax2.set_xlabel('Lateral (m)', fontsize=11)
    ax2.set_ylabel('Longitudinal (m)', fontsize=11)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.invert_xaxis()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_diversity_metrics(all_modes, selected, gt):
    """Compute trajectory diversity metrics."""
    metrics = {}
    if all_modes is None:
        return metrics

    num_modes = all_modes.shape[0]

    # 1. Average Pairwise Distance (APD) — measures spread of modes
    pairwise_dists = []
    for i in range(num_modes):
        for j in range(i + 1, num_modes):
            dist = np.linalg.norm(all_modes[i] - all_modes[j], axis=-1).mean()
            pairwise_dists.append(dist)
    metrics['avg_pairwise_dist'] = np.mean(pairwise_dists)

    # 2. Final Displacement Diversity (FDD) — spread of endpoints
    endpoints = all_modes[:, -1, :]  # (20, 2)
    endpoint_std = np.std(endpoints, axis=0)
    metrics['endpoint_std_x'] = endpoint_std[0]
    metrics['endpoint_std_y'] = endpoint_std[1]
    metrics['endpoint_spread'] = np.linalg.norm(endpoint_std)

    # 3. Coverage — how well modes cover the area around GT
    if gt is not None:
        gt_dist = np.linalg.norm(all_modes - gt[None, :, :], axis=-1)  # (20, 6)
        min_dist_per_step = gt_dist.min(axis=0)  # (6,)
        metrics['minADE'] = min_dist_per_step.mean()
        metrics['minFDE'] = gt_dist[:, -1].min()

        # Best mode index
        ade_per_mode = gt_dist.mean(axis=1)  # (20,)
        metrics['best_mode_idx'] = int(np.argmin(ade_per_mode))
        metrics['best_mode_ADE'] = ade_per_mode.min()

    # 4. Selected trajectory error (if available)
    if selected is not None and gt is not None:
        sel_dist = np.linalg.norm(selected - gt, axis=-1)
        metrics['selected_ADE'] = sel_dist.mean()
        metrics['selected_FDE'] = sel_dist[-1]

    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from mmcv.utils import Config, set_random_seed
    set_random_seed(args.seed)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    print(f'Building model and dataloader...')
    model, data_loader, dataset = build_model_and_dataloader(cfg, args.checkpoint)
    print(f'Model loaded. Dataset size: {len(dataset)}')

    # We need to patch the model to capture all 20 modes during inference
    # The key is in simple_test_pts where poses_reg has all 20 modes
    # but only the selected one is returned.
    # We'll patch the code to also store all modes.

    m = model.module if hasattr(model, 'module') else model
    plan_anchor = m.plan_anchor.detach().cpu().float().numpy()  # (20, 6, 2)

    os.environ['DEBUG_SHOW_PRED'] = '0'

    all_metrics = []
    sample_indices = list(range(0, min(len(dataset), args.num_samples * 50), 50))[:args.num_samples]

    import gc
    print(f'\nRunning inference on {len(sample_indices)} samples (CPU-GPU offloading)...')
    for count, idx in enumerate(sample_indices):
        data = dataset[idx]
        from mmcv.parallel import collate
        data_collated = collate([data], samples_per_gpu=1)

        torch.cuda.empty_cache()
        gc.collect()

        img_metas = data_collated['img_metas'][0].data[0] if hasattr(data_collated['img_metas'][0], 'data') else data_collated['img_metas'][0]

        # Phase 1: backbone on GPU → extract image features
        move_backbone_to_gpu(m)
        img_tensor = data_collated['img'][0]
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.unsqueeze(0)
        img_gpu = img_tensor.to(torch.bfloat16).cuda()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            img_feats = m.extract_feat(img_gpu)

        img_feats_cpu = img_feats.cpu()
        del img_gpu, img_feats
        move_backbone_to_cpu(m)
        torch.cuda.empty_cache()
        gc.collect()

        # Phase 2: LLM + heads on GPU → run planning inference
        move_rest_to_gpu(m)
        img_feats_gpu = img_feats_cpu.cuda()
        del img_feats_cpu

        data_gpu = {'img_feats': img_feats_gpu}
        data_gpu['img'] = img_tensor.to(torch.bfloat16).cuda()
        for k, v in data_collated.items():
            if k in ('img', 'img_metas'):
                continue
            if hasattr(v, 'data'):
                v = v.data
            if isinstance(v, list):
                v = v[0]
            if isinstance(v, torch.Tensor):
                if k in ('lidar2img', 'cam_intrinsic', 'ego_pose', 'ego_pose_inv', 'timestamp'):
                    data_gpu[k] = v.float().cuda()
                else:
                    data_gpu[k] = v.to(torch.bfloat16).cuda()
            else:
                data_gpu[k] = v

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            bbox_pts, generated_text, lane_results, metric_dict = m.simple_test_pts(
                img_metas, **data_gpu)

        move_rest_to_cpu(m)

        # Extract results
        selected = lane_results[0].get('ego_fut_preds', torch.zeros(6, 2))
        if isinstance(selected, torch.Tensor):
            selected = selected.cpu().float().numpy()

        gt = None
        if 'ego_fut_trajs' in data_gpu:
            gt_raw = data_gpu['ego_fut_trajs']
            if isinstance(gt_raw, torch.Tensor):
                gt = gt_raw[0, 0].cpu().float().cumsum(dim=-2).numpy()

        commands = ["LEFT", "RIGHT", "STRAIGHT", "LANE_FOLLOW", "CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"]
        cmd = data_gpu.get('command', None)
        if cmd is not None:
            cmd_idx = int(cmd.item()) if isinstance(cmd, torch.Tensor) else int(cmd)
            command = commands[cmd_idx] if cmd_idx < len(commands) else 'UNKNOWN'
        else:
            command = 'UNKNOWN'

        scene_token = img_metas[0].get('scene_token', 'unknown') if isinstance(img_metas, list) else img_metas.get('scene_token', 'unknown')
        frame_idx = img_metas[0].get('frame_idx', 0) if isinstance(img_metas, list) else img_metas.get('frame_idx', 0)

        # Get all 20 refined modes from lane_results (patched in orion.py)
        all_modes_tensor = lane_results[0].get('ego_fut_preds_all_modes', None)
        cls_scores_tensor = lane_results[0].get('ego_fut_cls_scores', None)

        if all_modes_tensor is not None:
            all_modes = all_modes_tensor[0].float().numpy() if all_modes_tensor.dim() == 4 else all_modes_tensor.float().numpy()
            cls_scores = cls_scores_tensor[0].float().numpy() if cls_scores_tensor is not None else None
            # Normalize cls_scores to [0, 1] via sigmoid
            if cls_scores is not None:
                cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))
        else:
            all_modes = plan_anchor
            cls_scores = None

        # Compute metrics
        metrics = compute_diversity_metrics(all_modes, selected, gt)
        metrics['command'] = command
        metrics['scene'] = str(scene_token)
        metrics['frame'] = frame_idx
        all_metrics.append(metrics)

        # Plot
        output_path = os.path.join(args.output_dir, f'sample_{count:03d}_{command}.png')
        plot_trajectories_bev(
            all_modes=all_modes,
            selected=selected,
            gt=gt,
            plan_anchor=plan_anchor,
            command=command,
            scene_token=scene_token,
            frame_idx=frame_idx,
            cls_scores=cls_scores,
            output_path=output_path,
        )
        print(f'  [{count+1}/{len(sample_indices)}] {command} | '
              f'selADE={metrics.get("selected_ADE", -1):.2f} | '
              f'minADE={metrics.get("minADE", -1):.2f} | '
              f'spread={metrics.get("endpoint_spread", -1):.2f}')

        # Free all GPU tensors from this iteration
        del data_gpu, data_collated, bbox_pts, generated_text, lane_results, metric_dict
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print('\n' + '=' * 60)
    print('DIVERSITY METRICS SUMMARY')
    print('=' * 60)
    valid = [m for m in all_metrics if 'avg_pairwise_dist' in m]
    if valid:
        print(f'  Avg Pairwise Distance:  {np.mean([m["avg_pairwise_dist"] for m in valid]):.3f} m')
        print(f'  Endpoint Spread:        {np.mean([m["endpoint_spread"] for m in valid]):.3f} m')
    if any('minADE' in m for m in all_metrics):
        valid_ade = [m for m in all_metrics if 'minADE' in m]
        print(f'  minADE (best of 20):    {np.mean([m["minADE"] for m in valid_ade]):.3f} m')
        print(f'  minFDE (best of 20):    {np.mean([m["minFDE"] for m in valid_ade]):.3f} m')
    if any('selected_ADE' in m for m in all_metrics):
        valid_sel = [m for m in all_metrics if 'selected_ADE' in m]
        print(f'  Selected ADE:           {np.mean([m["selected_ADE"] for m in valid_sel]):.3f} m')
        print(f'  Selected FDE:           {np.mean([m["selected_FDE"] for m in valid_sel]):.3f} m')
    print('=' * 60)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'diversity_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    print(f'\nMetrics saved to {metrics_path}')
    print(f'Visualizations saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
