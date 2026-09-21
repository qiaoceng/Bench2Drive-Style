"""StyleDrive: generate the ego-status text for every clip token listed in styletest.yaml.

For every token in the yaml's `tokens` list, locate the corresponding driving
log pkl, slide a window of `num_history + num_future` (=14) frames matching the
yaml scene filter, and pull clip frames 4-13 (the 10 future frames). Per-frame
ego velocity / acceleration / rotation-rate come from the pkl's `can_bus`
field; yaw is derived from `ego2global_rotation`.

can_bus layout (18 floats, verified against ego_dynamic_state):
    [gx, gy, gz, qw, qx, qy, qz, ax, ay, az, vx, vy, vz, wx, wy, wz, 0, 0]

Output: <clips-dir>/ego_status_in_text_enhanced/ego_status_in_text_enhanced.json
(one entry per clip token; same text format as the Bench2Drive ego status)

Usage:
    python -m style_labeling.styledrive.prepare.prepare_ego_context
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import yaml

from ...common.paths import SD_ROOT

OUT_FILE = "ego_status_in_text_enhanced.json"

AX, AY, AZ = 7, 8, 9
VX, VY, VZ = 10, 11, 12
WX, WY, WZ = 13, 14, 15


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def format_text(
    velocities: List[np.ndarray],
    accelerations: List[np.ndarray],
    rotation_rates: List[np.ndarray],
    yaws: List[float],
    total_displacement: float,
    num_objects: int,
    max_ax_value: float,
    max_ax_frame: int,
    max_dec_value: float,
    max_dec_frame: int,
    max_abs_ay_value: float,
    max_ay_frame: int,
) -> str:
    vels_str = "[" + ", ".join(repr(v) for v in velocities) + "]"
    accs_str = "[" + ", ".join(repr(a) for a in accelerations) + "]"
    rots_str = "[" + ", ".join(repr(r) for r in rotation_rates) + "]"
    yaws_str = "[" + ", ".join(repr(y) for y in yaws) + "]"
    return (
        "Driving context:\n"
        f"    - Ego velocity(x, y, z) from frame 0~9 : {vels_str}\n"
        f"    - Ego acceleration(x, y, z) from frame 0~9 : {accs_str}\n"
        f"    - Ego rotation rate(x, y, z) from frame 0~9 : {rots_str}\n"
        f"    - Ego yaw from frame 0~9 : {yaws_str}\n"
        f"    - Total displacement: {total_displacement:.1f} meters\n"
        f"    - Objects encountered: {num_objects} unique objects\n"
        f"    - [Pre-calculated] Max ax: {max_ax_value:.2e} m/s^2 at Frame {max_ax_frame}\n"
        f"    - [Pre-calculated] Max Deceleration: {max_dec_value:.2e} m/s^2 at Frame {max_dec_frame}\n"
        f"    - [Pre-calculated] Max |ay|: {max_abs_ay_value:.2e} m/s^2 at Frame {max_ay_frame}\n"
    )


def build_entry(token: str, clip_frames: List[Dict[str, Any]], img_dir_base: str) -> Dict[str, Any]:
    velocities: List[np.ndarray] = []
    accelerations: List[np.ndarray] = []
    rotation_rates: List[np.ndarray] = []
    yaws: List[float] = []
    positions: List[np.ndarray] = []
    object_track_tokens: Set[str] = set()
    object_categories = {"vehicle", "pedestrian"}
    max_distance_m = 50.0
    max_distance_sq = max_distance_m * max_distance_m

    for fr in clip_frames:
        cb = np.asarray(fr["can_bus"], dtype=np.float64)
        velocities.append(np.array([cb[VX], cb[VY], cb[VZ]]))
        accelerations.append(np.array([cb[AX], cb[AY], cb[AZ]]))
        rotation_rates.append(np.array([cb[WX], cb[WY], cb[WZ]]))

        rot = fr["ego2global_rotation"]
        yaws.append(quat_to_yaw(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])))

        trans = np.asarray(fr["ego2global_translation"], dtype=np.float64).reshape(-1)[:3]
        positions.append(trans)

        anns = fr.get("anns") or {}
        names = anns.get("gt_names")
        tracks = anns.get("track_tokens")
        boxes = anns.get("gt_boxes")
        if names is not None and tracks is not None and boxes is not None:
            boxes_arr = np.asarray(boxes)
            for idx, (name, track) in enumerate(zip(names, tracks)):
                if name not in object_categories:
                    continue
                bx, by = float(boxes_arr[idx, 0]), float(boxes_arr[idx, 1])
                if bx * bx + by * by <= max_distance_sq:
                    object_track_tokens.add(track)

    total_disp = float(
        sum(np.linalg.norm(positions[k] - positions[k - 1]) for k in range(1, len(positions)))
    )

    ax_arr = np.array([a[0] for a in accelerations])
    ay_arr = np.array([a[1] for a in accelerations])
    max_ax_frame = int(np.argmax(ax_arr))
    max_ax_value = float(ax_arr[max_ax_frame])
    max_dec_frame = int(np.argmin(ax_arr))
    max_dec_value = float(ax_arr[max_dec_frame])
    max_ay_frame = int(np.argmax(np.abs(ay_arr)))
    max_abs_ay_value = float(abs(ay_arr[max_ay_frame]))

    text = format_text(
        velocities,
        accelerations,
        rotation_rates,
        yaws,
        total_disp,
        len(object_track_tokens),
        max_ax_value,
        max_ax_frame,
        max_dec_value,
        max_dec_frame,
        max_abs_ay_value,
        max_ay_frame,
    )

    videos = [
        f"{img_dir_base}/{token}/{k:05d}.jpg" for k in range(4, 14)
    ]

    return {
        "role": "user",
        "clip_token": token,
        "content": [
            {"type": "video", "video": videos, "sample_fps": "2"},
            {"type": "text", "text": text},
        ],
    }


def process_log(
    log_pkl_path: Path,
    target_tokens: Set[str],
    num_history: int,
    num_future: int,
    frame_interval: int,
    has_route: bool,
    img_dir_base: str,
) -> List[Dict[str, Any]]:
    with open(log_pkl_path, "rb") as f:
        scene_dict_list: List[Dict[str, Any]] = pickle.load(f)

    num_frames = num_history + num_future
    entries: List[Dict[str, Any]] = []
    seen_in_log: Set[str] = set()
    n = len(scene_dict_list)

    for i in range(0, n, frame_interval):
        frame_list = scene_dict_list[i : i + num_frames]
        if len(frame_list) < num_frames:
            continue
        if has_route and len(frame_list[num_history - 1].get("roadblock_ids") or []) == 0:
            continue
        token = frame_list[num_history - 1]["token"]
        if token not in target_tokens or token in seen_in_log:
            continue
        seen_in_log.add(token)
        clip_frames = frame_list[num_history:num_history + num_future]
        entries.append(build_entry(token, clip_frames, img_dir_base))

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", default=str(SD_ROOT / "navsim_logs" / "test"), help="NAVSIM test log pkls")
    parser.add_argument("--scene-filter", default=str(SD_ROOT / "styletest.yaml"),
                        help="StyleDrive scene filter (navsim/planning/script/config/common/train_test_split/scene_filter/styletest.yaml)")
    parser.add_argument("--clips-dir", default=str(SD_ROOT / "StyleClips"))
    args = parser.parse_args()
    PKL_DIR = args.logs_dir
    OUT_DIR = os.path.join(args.clips_dir, "ego_status_in_text_enhanced")
    img_dir_base = os.path.join(args.clips_dir, "multi_view_images")

    with open(args.scene_filter, "r") as f:
        cfg = yaml.safe_load(f)

    num_history = int(cfg.get("num_history_frames", 4))
    num_future = int(cfg.get("num_future_frames", 10))
    frame_interval = int(cfg.get("frame_interval", 1))
    has_route = bool(cfg.get("has_route", True))
    log_names: List[str] = list(cfg.get("log_names") or [])
    tokens: Set[str] = set(cfg.get("tokens") or [])

    assert num_history == 4 and num_future == 10, "expected 4 history / 10 future frames"
    print(f"yaml: {len(log_names)} logs, {len(tokens)} tokens, frame_interval={frame_interval}")

    os.makedirs(OUT_DIR, exist_ok=True)

    pkl_dir = Path(PKL_DIR)
    all_entries: List[Dict[str, Any]] = []
    matched_tokens: Set[str] = set()

    for log_name in log_names:
        pkl_path = pkl_dir / f"{log_name}.pkl"
        if not pkl_path.exists():
            print(f"  [missing pkl] {pkl_path}")
            continue
        entries = process_log(
            pkl_path, tokens, num_history, num_future, frame_interval, has_route, img_dir_base
        )
        for e in entries:
            matched_tokens.add(e["clip_token"])
        all_entries.extend(entries)
        print(f"  {log_name}: +{len(entries)} (running total {len(all_entries)})")

    missing = tokens - matched_tokens
    print(f"matched {len(matched_tokens)}/{len(tokens)} tokens; {len(missing)} missing")

    out_path = os.path.join(OUT_DIR, OUT_FILE)
    with open(out_path, "w") as f:
        json.dump(all_entries, f, indent=2)
    print(f"wrote {out_path} ({len(all_entries)} entries)")


if __name__ == "__main__":
    main()
