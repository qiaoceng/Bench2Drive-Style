"""
StyleDrive (NAVSIM test split): render a 2x3 surround-view composite for every frame of the
14-frame window (4 history + 10 future) around each styletest clip token.

Output: <clips-dir>/multi_view_images/<clip_token>/<00000..00013>.jpg

Usage:
    python -m style_labeling.styledrive.prepare.prepare_multiview
"""
import argparse
import pickle
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from PIL import Image
from tqdm import tqdm

from ...common.paths import SD_ROOT

# Each entry is (navsim_cam_key, display_label).
CAMERAS_2x3 = [
    [("CAM_F0", "CAM_FRONT"), ("CAM_L0", "CAM_FRONT_LEFT"), ("CAM_R0", "CAM_FRONT_RIGHT")],
    [("CAM_B0", "CAM_BACK"),  ("CAM_L2", "CAM_BACK_LEFT"),  ("CAM_R2", "CAM_BACK_RIGHT")],
]


def load_scene_filter(yaml_path):
    """Load log_names, tokens, num_history_frames, num_future_frames from the SceneFilter yaml."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return {
        "log_names": cfg["log_names"],
        "tokens": cfg["tokens"],
        "num_history_frames": cfg["num_history_frames"],
        "num_future_frames": cfg["num_future_frames"],
    }


def find_clip_frames(scene_dict_list, rep_token, num_history, num_future):
    """Locate the (num_history+num_future)-frame clip around the representative token in the log pkl."""
    tokens = [d["token"] for d in scene_dict_list]
    if rep_token not in tokens:
        raise ValueError(f"Token {rep_token} not found in pkl")
    rep_idx = tokens.index(rep_token)
    start = rep_idx - (num_history - 1)
    end = rep_idx + num_future + 1
    if start < 0 or end > len(scene_dict_list):
        raise ValueError(
            f"Clip window pkl[{start}:{end}] out of bounds (pkl_len={len(scene_dict_list)})"
        )
    return scene_dict_list[start:end]


def make_multiview_image(frame, output_path, cameras_grid, sensor_root):
    """Render a 2x3 multi-view composite for a single frame and save as jpg."""
    nrow = len(cameras_grid)
    ncol = len(cameras_grid[0])
    fig, axes = plt.subplots(nrow, ncol, figsize=(25, 10))

    for r in range(nrow):
        for c in range(ncol):
            cam_key, cam_label = cameras_grid[r][c]
            ax = axes[r][c]
            cam_info = frame["cams"].get(cam_key)
            if cam_info is None:
                ax.text(0.5, 0.5, f"{cam_label}\nNot available", ha="center", va="center")
                ax.axis("off")
                continue
            img_path = os.path.join(sensor_root, cam_info["data_path"])
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(cam_label)
                ax.axis("off")
            except FileNotFoundError:
                ax.text(0.5, 0.5, f"{cam_label}\nFile not found", ha="center", va="center")
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=100, format="jpg")
    plt.close(fig)


def build_token_to_log(pkl_dir, log_names, tokens):
    """Walk each log pkl once and resolve which log each token lives in."""
    token_set = set(tokens)
    token_to_log = {}
    for log in tqdm(log_names, desc="Indexing tokens"):
        pkl_path = os.path.join(pkl_dir, f"{log}.pkl")
        if not os.path.exists(pkl_path):
            print(f"[WARN] missing pkl: {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            scene_dict_list = pickle.load(f)
        for d in scene_dict_list:
            t = d["token"]
            if t in token_set and t not in token_to_log:
                token_to_log[t] = log
    return token_to_log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", default=str(SD_ROOT / "navsim_logs" / "test"), help="NAVSIM test log pkls")
    parser.add_argument("--sensor-dir", default=str(SD_ROOT / "sensor_blobs" / "test"), help="NAVSIM test sensor blobs")
    parser.add_argument("--scene-filter", default=str(SD_ROOT / "styletest.yaml"),
                        help="StyleDrive scene filter (navsim/planning/script/config/common/train_test_split/scene_filter/styletest.yaml)")
    parser.add_argument("--clips-dir", default=str(SD_ROOT / "StyleClips"))
    args = parser.parse_args()
    pkl_dir = args.logs_dir
    multi_view_dir = os.path.join(args.clips_dir, "multi_view_images")

    cfg = load_scene_filter(args.scene_filter)
    log_names = cfg["log_names"]
    tokens = cfg["tokens"]
    num_history = cfg["num_history_frames"]
    num_future = cfg["num_future_frames"]
    clip_len = num_history + num_future
    print(f"yaml: {len(log_names)} log_names, {len(tokens)} tokens, clip_len={clip_len}")

    # Resolve token -> log, then group tokens by log (so each pkl is loaded once).
    token_to_log = build_token_to_log(pkl_dir, log_names, tokens)
    missing = [t for t in tokens if t not in token_to_log]
    if missing:
        print(f"[WARN] {len(missing)} tokens not found in any pkl; first few: {missing[:5]}")

    tokens_by_log = defaultdict(list)
    for t in tokens:
        if t in token_to_log:
            tokens_by_log[token_to_log[t]].append(t)

    total_clips = sum(len(v) for v in tokens_by_log.values())
    print(f"Will generate {total_clips} clips x {clip_len} frames = {total_clips * clip_len} images")

    pbar = tqdm(total=total_clips, desc="Clips")
    for log_name, log_tokens in tokens_by_log.items():
        with open(os.path.join(pkl_dir, f"{log_name}.pkl"), "rb") as f:
            scene_dict_list = pickle.load(f)

        for rep_token in log_tokens:
            scene_dir = os.path.join(multi_view_dir, rep_token)

            # Skip if this clip already has a full set of jpgs.
            if os.path.isdir(scene_dir):
                existing = [n for n in os.listdir(scene_dir) if n.endswith(".jpg")]
                if len(existing) >= clip_len:
                    pbar.update(1)
                    continue

            try:
                clip = find_clip_frames(scene_dict_list, rep_token, num_history, num_future)
            except ValueError as e:
                print(f"[SKIP] {rep_token} ({log_name}): {e}")
                pbar.update(1)
                continue

            os.makedirs(scene_dir, exist_ok=True)
            for i, frame in enumerate(clip):
                output_path = os.path.join(scene_dir, f"{i:05d}.jpg")
                make_multiview_image(frame, output_path, CAMERAS_2x3, args.sensor_dir)

            pbar.update(1)
    pbar.close()
    print(f"Done. Images saved under: {multi_view_dir}")


if __name__ == "__main__":
    main()