#!/usr/bin/env python3
# ==============================================================================
# [Bench2Drive-Style] NEW FILE -- training-pickle generation
# Author: YinXuan1223   (carried over from the ORION_NYCUACM_version fork)
#
# Merge the VLM clip-level driving-style labels into the info pickle, adding a
# per-frame  'driving_style'  float in [-1, 1]  (-1 conservative / 0 normal /
# +1 aggressive) to every frame.  b2d_infos_train.pkl -> b2d_infos_train_with_style.pkl.
# Clip labels are interpolated over gaps, Gaussian-smoothed along the clip index
# (--sigma, default 4) and written to each clip's two centre frames, averaging where
# clips overlap; frames with no label get 0.0.  This is the file that makes the
# style-conditioned planner trainable.
# ==============================================================================
"""
Merge Gaussian-smoothed style scores into B2D training PKL (Method G: mid-2 frames).

Clip structure:
    - Each clip contains 10 frames at 2Hz (sample_interval=5)
    - Clip i covers 10Hz frame indices: [i, i+5, i+10, ..., i+45]
    - Clip stride = 1 (in 10Hz), so clip index == 10Hz starting frame

Pipeline:
    1. Load clip-level labels, fill missing/unknown via linear interpolation
    2. Gaussian smooth at clip level (sigma)
    3. Map to 10Hz frames using Method G: each clip contributes to its
       two center 2Hz frames (offsets +20, +25), overlapping clips averaged
    4. Interpolate any remaining gaps, clip to [-1, 1]
    5. Scenes with contiguous missing clips > max_gap are flagged;
       affected 10Hz frames are excluded (set to 0.0)

Usage:
    python prepare_style_pkl.py \
        --style-dir /path/to/StyleResults/version \
        --pkl-in /path/to/b2d_infos_train.pkl \
        --pkl-out /path/to/b2d_infos_train_with_style.pkl \
        --sigma 4

Each frame in the output PKL gets a new field:
    'driving_style': float in [-1, 1]
        -1 = conservative, 0 = normal, 1 = aggressive

Frames without style annotation get driving_style = 0.0 (neutral).
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Set, Tuple

LABEL_MAP = {"aggressive": 1.0, "normal": 0.0, "conservative": -1.0}
def interpolate_nans(arr):
    # type: (np.ndarray) -> np.ndarray
    """Fill NaN values via linear interpolation; edge NaNs use nearest valid."""
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    valid = ~nans
    if not valid.any():
        return np.zeros_like(arr)
    xp = np.where(valid)[0]
    fp = arr[valid]
    arr_filled = np.interp(np.arange(len(arr)), xp, fp)
    return arr_filled



def load_and_smooth(scene_dir, num_10hz_frames, sigma):
    # type: (str, int, float) -> Tuple[Dict[int, float], int, List[str]]
    """
    Load clip-level labels, handle missing/unknown, Gaussian smooth,
    then map to 10Hz frames using Method G (mid-2: offsets +20, +25).

    Returns:
        scores: {10hz_frame_idx: score} for valid frames
        coverage_count: number of 10Hz frames with valid style
        warnings: list of warning messages
    """
    json_path = os.path.join(scene_dir, "parsed_reasons_vllm.json")
    with open(json_path) as f:
        clips = json.load(f)

    warnings = []
    entries = sorted(clips, key=lambda x: x["index"])
    max_clip_idx = entries[-1]["index"]
    num_clips_total = max_clip_idx + 1

    # Build full clip array; track which indices exist in JSON vs which have valid labels
    clip_raw = np.full(num_clips_total, np.nan, dtype=np.float64)
    json_indices = set()  # indices that appear in the JSON file at all
    unknown_count = 0
    for e in entries:
        ci = e["index"]
        json_indices.add(ci)
        label = e["content"].get("Label", "unknown")
        if label in LABEL_MAP:
            clip_raw[ci] = LABEL_MAP[label]
        else:
            unknown_count += 1
            # unknown labels → NaN, will be interpolated (NOT treated as missing)

    if unknown_count > 0:
        warnings.append("  {} unknown labels (interpolated)".format(unknown_count))

    # Check for truly missing clip indices (not in JSON at all, distinct from unknown labels)
    # Clips starting from index > 0 is normal (some scenes start at 1); only check within range
    first_idx = entries[0]["index"]
    truly_missing = sorted(set(range(first_idx, max_clip_idx + 1)) - json_indices)
    cutoff_frame = num_10hz_frames  # default: use all frames
    if truly_missing:
        first_missing = truly_missing[0]
        warnings.append("  {} missing clip indices (not in JSON), first at clip {}".format(
            len(truly_missing), first_missing))
        warnings.append("  TRUNCATING: discarding clips [{}, {}] and all 10Hz frames >= {}".format(
            first_missing, max_clip_idx, first_missing))
        clip_raw = clip_raw[:first_missing]
        num_clips_total = first_missing
        cutoff_frame = first_missing

    # Interpolate NaN values (unknown labels) before Gaussian smoothing
    clip_filled = interpolate_nans(clip_raw)

    # Gaussian smooth at clip level
    clip_smoothed = gaussian_filter1d(clip_filled, sigma=sigma)
    clip_smoothed = np.clip(clip_smoothed, -1.0, 1.0)

    # Method G: map to 10Hz frames using mid-2 (offsets +20, +25)
    frame_sum = np.zeros(num_10hz_frames, dtype=np.float64)
    frame_cnt = np.zeros(num_10hz_frames, dtype=np.float64)
    for ci in range(num_clips_total):
        for offset in [20, 25]:
            f = ci + offset
            if 0 <= f < cutoff_frame:
                frame_sum[f] += clip_smoothed[ci]
                frame_cnt[f] += 1

    # Average where multiple clips contribute
    frame_scores = np.full(cutoff_frame, np.nan, dtype=np.float64)
    mask = frame_cnt[:cutoff_frame] > 0
    frame_scores[mask] = frame_sum[:cutoff_frame][mask] / frame_cnt[:cutoff_frame][mask]

    # Interpolate remaining gaps (head frames not covered by any clip's mid-2)
    frame_scores = interpolate_nans(frame_scores)
    frame_scores = np.clip(frame_scores, -1.0, 1.0)

    # Build result — only frames before cutoff
    result = {}
    for f in range(cutoff_frame):
        result[f] = float(frame_scores[f])

    excluded = num_10hz_frames - cutoff_frame
    return result, cutoff_frame, excluded, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--style-dir", required=True,
                        help="StyleResults version directory with per-scene subdirs")
    parser.add_argument("--pkl-in", required=True,
                        help="Input b2d_infos_train.pkl")
    parser.add_argument("--pkl-out", required=True,
                        help="Output PKL with driving_style field")
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="Gaussian smoothing sigma (default: 4)")
    args = parser.parse_args()

    print("Loading PKL from {}".format(args.pkl_in))
    with open(args.pkl_in, "rb") as f:
        infos = pickle.load(f)

    # Build scene → frame count mapping
    scene_frame_counts = {}  # type: Dict[str, int]
    for info in infos:
        folder = info["folder"]
        idx = info["frame_idx"]
        if folder not in scene_frame_counts or idx + 1 > scene_frame_counts[folder]:
            scene_frame_counts[folder] = idx + 1

    # Load and smooth style annotations
    print("Loading style annotations from {} (sigma={}, method=G mid-2)".format(
        args.style_dir, args.sigma))
    style_lookup = {}  # type: Dict[Tuple[str, int], float]
    scene_dirs = sorted([
        d for d in os.listdir(args.style_dir)
        if os.path.isdir(os.path.join(args.style_dir, d))
    ])

    total_covered = 0
    total_frames = 0
    total_excluded = 0
    scenes_with_issues = []

    for scene_name in scene_dirs:
        folder_key = "v1/{}".format(scene_name)
        num_frames = scene_frame_counts.get(folder_key, 0)
        if num_frames == 0:
            print("  WARNING: {} not found in PKL, skipping".format(scene_name))
            continue

        scene_path = os.path.join(args.style_dir, scene_name)
        scores, coverage, excluded, scene_warnings = load_and_smooth(
            scene_path, num_frames, args.sigma)

        for frame_idx, score in scores.items():
            style_lookup[(folder_key, frame_idx)] = score

        total_covered += coverage
        total_frames += num_frames
        total_excluded += excluded

        vals = list(scores.values()) if scores else [0.0]
        status = "OK" if not scene_warnings else "TRUNCATED" if excluded > 0 else "ISSUES"
        print("  {} [{}]: {} frames, {}/{} covered, {} excluded, "
              "range [{:.2f}, {:.2f}]".format(
                  scene_name, status, num_frames, coverage, num_frames,
                  excluded, min(vals), max(vals)))
        for w in scene_warnings:
            print(w)

    print("\nSummary: {} scenes, {}/{} frames covered, {} excluded".format(
        len(scene_dirs), total_covered, total_frames, total_excluded))

    # Inject driving_style into PKL
    annotated_count = 0
    for info in infos:
        key = (info["folder"], info["frame_idx"])
        if key in style_lookup:
            info["driving_style"] = style_lookup[key]
            annotated_count += 1
        else:
            info["driving_style"] = 0.0

    print("\nTotal PKL frames: {}".format(len(infos)))
    print("Frames with style: {}".format(annotated_count))
    print("Frames without style (default 0.0): {}".format(len(infos) - annotated_count))

    # Save
    print("\nSaving to {}".format(args.pkl_out))
    with open(args.pkl_out, "wb") as f:
        pickle.dump(infos, f)
    print("Done.")

    # Verification
    styled = np.array([info["driving_style"] for info in infos if info["driving_style"] != 0.0])
    if len(styled) > 0:
        print("\nVerification (non-zero styles):")
        print("  Count: {}".format(len(styled)))
        print("  Mean:  {:.3f}".format(styled.mean()))
        print("  Std:   {:.3f}".format(styled.std()))
        print("  Min:   {:.3f}".format(styled.min()))
        print("  Max:   {:.3f}".format(styled.max()))


if __name__ == "__main__":
    main()
