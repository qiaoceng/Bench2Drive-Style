"""
StyleDrive (NAVSIM test split): draw one BEV image per frame for the 10 future frames
(clip positions 4..13) of each styletest clip token, in the same style as the Bench2Drive BEVs.
Needs nuplan-devkit (for the map API).

Output: <clips-dir>/BEV_images/<clip_token>/<00000..00009>.jpg

Usage:
    python -m style_labeling.styledrive.prepare.prepare_bev
"""
import argparse
import os
import hashlib
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from pyquaternion import Quaternion
from tqdm import tqdm

from ...common.paths import SD_ROOT

# User-requested BEV slice within the (num_history + num_future)-frame clip.
CLIP_START_POS = 4   # first BEV frame (inclusive)
CLIP_END_POS = 13    # last  BEV frame (inclusive)
REF_POS = 4          # BEV origin / window center = ego at this position

# B2D BEV settings
MAX_DISTANCE = 50.0
FIGSIZE = (14, 14)
EGO_LENGTH = 4.5
EGO_WIDTH = 2.0

# Colors / styles -- copied verbatim from bench2drive/prepare/prepare_bev.py.
LANE_COLOR_MAP = {"Broken": "#888888", "Solid": "#000000", "SolidSolid": "#CC8800", "Center": "#FF5500", "NONE": "#000000"}
LANE_LINESTYLE_MAP = {"Broken": "--", "Solid": "-", "SolidSolid": "-", "Center": "-", "NONE": "-"}


def get_color_for_class(class_name):
    """navsim/nuPlan class -> color (colors borrowed from bench2drive/prepare/prepare_bev.py)."""
    n = class_name.lower()
    if n == "vehicle":           return "#0F29E9"   # 'Car' blue (navsim does not split car/van/truck/bus)
    if n == "pedestrian":        return "#DAE93E"   # yellow
    if n == "bicycle":           return "#741BBE"   # purple
    if n == "traffic_cone":      return "#FFD700"   # gold (B2D 'Static Prop')
    if n == "generic_object":    return "#888888"   # gray (small unknown stuff)
    if n.startswith("static."):  return "#FFD700"   # B2D fallback
    if n.startswith("traffic."): return "#FF0000"   # B2D fallback
    return "#0F29E9"


def str_to_color(s):
    return "#" + hashlib.md5(s.encode("utf-8")).hexdigest()[:6]


def yaw_from_quat(q):
    """pkl ego2global_rotation = [w, x, y, z]. Returns yaw in radians."""
    return Quaternion(*q).yaw_pitch_roll[0]


def transform_world_to_ref_xy(xy_world, ref_xy, ref_yaw):
    """World XY -> ref-ego-local XY. 2D, yaw-only (ignores z / pitch / roll).
    Using a 2D transform avoids the bug where setting z=0 in a 4x4 inverse projects
    points off-axis because the ref ego sits at a non-zero world elevation."""
    xy_world = np.atleast_2d(xy_world).astype(np.float64)
    dx = xy_world[:, 0] - ref_xy[0]
    dy = xy_world[:, 1] - ref_xy[1]
    c, s = np.cos(-ref_yaw), np.sin(-ref_yaw)
    return np.stack([c * dx - s * dy, s * dx + c * dy], axis=1)


def rotate_xy_2d(xy, theta):
    """Rotate a single (x, y) by theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


# Display rotation: navsim/nuPlan ego frame uses +X=forward, +Y=left.
# We rotate the displayed coords 90deg CCW so that ego's forward (+X) points UP on screen
# (and ego's left (+Y) points to screen LEFT).  (x, y) -> (-y, x) ; yaws get +pi/2.
def to_bev_display(xy):
    """(N, 2) or (2,) in ego-local -> rotated for BEV display."""
    xy = np.asarray(xy)
    if xy.ndim == 1:
        return np.array([-xy[1], xy[0]])
    return np.stack([-xy[:, 1], xy[:, 0]], axis=1)


YAW_OFFSET = np.pi / 2  # add this to every plotted yaw because of the display rotation above.


def find_clip(scene_dict_list, rep_token, num_history, num_future):
    tokens = [d["token"] for d in scene_dict_list]
    rep_idx = tokens.index(rep_token)
    start = rep_idx - (num_history - 1)
    end = rep_idx + num_future + 1
    return start, end


def collect_clip_state(scene_dict_list, clip_pkl_indices, ref_pkl_idx):
    """Build coordinate transforms, ego trajectory, object trajectories in ref-ego frame."""
    ref_frame = scene_dict_list[ref_pkl_idx]
    ref_xy = np.array(ref_frame["ego2global_translation"])[:2]
    ref_yaw = yaw_from_quat(ref_frame["ego2global_rotation"])

    ego_positions = np.zeros((len(clip_pkl_indices), 2))
    ego_yaws = np.zeros(len(clip_pkl_indices))

    object_trajectories = {}  # track_token -> dict

    for clip_frame_num, pkl_idx in enumerate(clip_pkl_indices):
        frame_i = scene_dict_list[pkl_idx]
        yaw_i = yaw_from_quat(frame_i["ego2global_rotation"])
        ego_xy_world = np.array(frame_i["ego2global_translation"])[:2]

        # Ego pose in ref-ego frame, then rotated to BEV display.
        ego_xy_ref = transform_world_to_ref_xy(ego_xy_world, ref_xy, ref_yaw)[0]
        ego_positions[clip_frame_num] = to_bev_display(ego_xy_ref)
        ego_yaws[clip_frame_num] = (yaw_i - ref_yaw) + YAW_OFFSET

        # Boxes are in frame_i's ego-local frame; transform to ref-ego frame:
        # rotate by rel_yaw, then add frame_i's origin in ref-local.
        rel_yaw = yaw_i - ref_yaw
        anns = frame_i["anns"]
        for box, name, tok in zip(anns["gt_boxes"], anns["gt_names"], anns["track_tokens"]):
            xy_local_ref = rotate_xy_2d(box[:2], rel_yaw) + ego_xy_ref
            if np.linalg.norm(xy_local_ref) > MAX_DISTANCE * 2:
                continue

            xy_disp = to_bev_display(xy_local_ref)
            length, width = box[3], box[4]
            box_yaw_disp = box[6] + rel_yaw + YAW_OFFSET

            entry = object_trajectories.setdefault(
                tok,
                {"positions": [], "yaws": [], "widths": [], "lengths": [], "name": name, "clip_frame_nums": []},
            )
            entry["positions"].append(xy_disp)
            entry["yaws"].append(box_yaw_disp)
            entry["widths"].append(width)
            entry["lengths"].append(length)
            entry["clip_frame_nums"].append(clip_frame_num)

    return ego_positions, ego_yaws, object_trajectories, (ref_xy, ref_yaw)


def fetch_map_data(map_api, ref_frame, ref_xy_yaw):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    """Query nuPlan map around ref-ego and transform geometries to ref-ego local frame."""
    ref_xy, ref_yaw = ref_xy_yaw
    ex, ey = ref_frame["ego2global_translation"][:2]
    # Query a radius that fully covers the ±MAX_DISTANCE display square (corners at sqrt(2)*MAX_DISTANCE).
    map_radius = MAX_DISTANCE * np.sqrt(2)
    objs = map_api.get_proximal_map_objects(
        point=Point2D(ex, ey),
        radius=map_radius,
        layers=[
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.INTERSECTION,
        ],
    )

    # Lanes: nuPlan doesn't expose Broken/Solid line types per boundary, but each boundary
    # object has a unique id and adjacent lanes share the same boundary object. So:
    #   - boundary used by 2 lanes/connectors  -> lane divider -> 'Broken' (dashed gray)
    #   - boundary used by 1 lane/connector    -> road edge    -> 'Solid'  (black)
    #   - baseline_path                         -> 'Center'    (orange)
    from collections import Counter
    boundary_use_count: Counter = Counter()
    for lane in objs[SemanticMapLayer.LANE]:
        boundary_use_count[lane.left_boundary.id]  += 1
        boundary_use_count[lane.right_boundary.id] += 1
    for lc in objs[SemanticMapLayer.LANE_CONNECTOR]:
        boundary_use_count[lc.left_boundary.id]  += 1
        boundary_use_count[lc.right_boundary.id] += 1

    drawn_boundary_ids = set()  # avoid drawing the same shared boundary twice

    def _add_local(pts_world_xy, label):
        pts_local = transform_world_to_ref_xy(pts_world_xy, ref_xy, ref_yaw)
        nearby_lanes.append((to_bev_display(pts_local), label))

    def _add_boundary(bd):
        if bd.id in drawn_boundary_ids:
            return
        drawn_boundary_ids.add(bd.id)
        label = "Broken" if boundary_use_count[bd.id] >= 2 else "Solid"
        _add_local(np.array(bd.linestring.coords)[:, :2], label)

    nearby_lanes = []  # list of (pts_display (N,2), lane_type str)
    for lane in objs[SemanticMapLayer.LANE]:
        _add_local(np.array(lane.baseline_path.linestring.coords)[:, :2], "Center")
        _add_boundary(lane.left_boundary)
        _add_boundary(lane.right_boundary)
    for lc in objs[SemanticMapLayer.LANE_CONNECTOR]:
        _add_local(np.array(lc.baseline_path.linestring.coords)[:, :2], "Center")
        _add_boundary(lc.left_boundary)
        _add_boundary(lc.right_boundary)

    # Triggers: B2D had only TrafficLight + StopSign + hashed fallbacks.
    #   - STOP_LINE  -> 'StopSign'  (red fill)
    #   - CROSSWALK  -> hashed color (B2D-style fallback)
    nearby_triggers = []
    for sl in objs[SemanticMapLayer.STOP_LINE]:
        pts = np.array(sl.polygon.exterior.coords)[:, :2]
        nearby_triggers.append((to_bev_display(transform_world_to_ref_xy(pts, ref_xy, ref_yaw)), "StopSign"))
    for cw in objs[SemanticMapLayer.CROSSWALK]:
        pts = np.array(cw.polygon.exterior.coords)[:, :2]
        nearby_triggers.append((to_bev_display(transform_world_to_ref_xy(pts, ref_xy, ref_yaw)), "Crosswalk"))

    return nearby_lanes, nearby_triggers


def draw_per_frame_bev(
    ego_positions, ego_yaws, object_trajectories, nearby_lanes, nearby_triggers,
    output_dir,
):
    """Draw one BEV per clip frame (mirrors create_per_frame_bev from B2D)."""
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    num_frames = len(ego_positions)

    for current_i in range(num_frames):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        # --- Map ---
        for lane_pts, lane_type in nearby_lanes:
            color = LANE_COLOR_MAP.get(lane_type, "#888888")
            ls = LANE_LINESTYLE_MAP.get(lane_type, "-")
            ax.plot(lane_pts[:, 0], lane_pts[:, 1], color=color, linewidth=1.2, alpha=0.6, linestyle=ls, zorder=1)

        for trig_pts, trig_type in nearby_triggers:
            if trig_type == "TrafficLight":
                color = "#008800"
            elif trig_type == "StopSign":
                color = "#CC0000"
            else:
                color = str_to_color(trig_type)
            ax.fill(trig_pts[:, 0], trig_pts[:, 1], color=color, alpha=0.25, zorder=2)
            ax.plot(
                np.append(trig_pts[:, 0], trig_pts[0, 0]),
                np.append(trig_pts[:, 1], trig_pts[0, 1]),
                color=color, linewidth=1.0, alpha=0.6, zorder=2,
            )

        # --- Ego past trajectory (BEV-frame 0 .. current) ---
        if current_i > 0:
            ax.plot(
                ego_positions[:current_i + 1, 0], ego_positions[:current_i + 1, 1],
                color="green", linewidth=4, linestyle="--", alpha=0.8, zorder=5,
            )

        # --- Ego rectangle ---
        ego_rect = Rectangle(
            (-EGO_LENGTH / 2, -EGO_WIDTH / 2), EGO_LENGTH, EGO_WIDTH,
            linewidth=2, edgecolor="green", facecolor="green", alpha=0.5,
        )
        t = (
            Affine2D().rotate(ego_yaws[current_i])
            .translate(ego_positions[current_i, 0], ego_positions[current_i, 1])
            + ax.transData
        )
        ego_rect.set_transform(t)
        ax.add_patch(ego_rect)
        ax.scatter(
            ego_positions[current_i, 0], ego_positions[current_i, 1],
            color="green", s=100, marker="*", edgecolors="white", linewidths=2, zorder=11,
        )

        # --- Other objects ---
        for tok, obj_data in object_trajectories.items():
            positions = np.array(obj_data["positions"])
            clip_frame_nums = obj_data["clip_frame_nums"]

            if current_i not in clip_frame_nums:
                continue

            valid_indices = [i for i, f_num in enumerate(clip_frame_nums) if f_num <= current_i]
            if not valid_indices:
                continue

            cur_obj_positions = positions[valid_indices]
            color = get_color_for_class(obj_data["name"])

            if len(cur_obj_positions) >= 2:
                ax.plot(
                    cur_obj_positions[:, 0], cur_obj_positions[:, 1],
                    color=color, linewidth=3.5, linestyle="--", alpha=0.9, zorder=5,
                )

            idx_at_current = clip_frame_nums.index(current_i)
            l = obj_data["lengths"][idx_at_current]
            w = obj_data["widths"][idx_at_current]
            yaw = obj_data["yaws"][idx_at_current]
            pos = positions[idx_at_current]

            rect = Rectangle((-l / 2, -w / 2), l, w, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.4)
            rect.set_transform(Affine2D().rotate(yaw).translate(pos[0], pos[1]) + ax.transData)
            ax.add_patch(rect)
            ax.scatter(pos[0], pos[1], color=color, s=60, marker="o", edgecolors="white", linewidths=1, alpha=0.9, zorder=9)

        ax.set_xlim(-MAX_DISTANCE, MAX_DISTANCE)
        ax.set_ylim(-MAX_DISTANCE, MAX_DISTANCE)
        ax.set_xlabel("X (m)", fontsize=12, color="black")
        ax.set_ylabel("Y (m)", fontsize=12, color="black")
        ax.set_title(f"Per Frame BEV - Frame {current_i}", fontsize=14, color="black")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, color="gray")

        # --- Legend ---
        legend_elements = []
        legend_elements.append(plt.Line2D([0], [0], color="white", linewidth=0, label="── Map ──"))
        lane_types_shown = {t for _, t in nearby_lanes}
        if "Broken" in lane_types_shown:     legend_elements.append(plt.Line2D([0], [0], color="#888888", linewidth=2, linestyle="--", label="Broken Lane"))
        if "Solid" in lane_types_shown:      legend_elements.append(plt.Line2D([0], [0], color="#000000", linewidth=2, linestyle="-",  label="Solid Lane"))
        if "SolidSolid" in lane_types_shown: legend_elements.append(plt.Line2D([0], [0], color="#CC8800", linewidth=2, linestyle="-",  label="Double Solid"))
        if "Center" in lane_types_shown:     legend_elements.append(plt.Line2D([0], [0], color="#FF5500", linewidth=2, linestyle="-",  label="Center Line"))

        trigger_types_shown = {t for _, t in nearby_triggers}
        for tt in trigger_types_shown:
            if   tt == "TrafficLight": color = "#008800"
            elif tt == "StopSign":     color = "#CC0000"
            else:                      color = str_to_color(tt)
            legend_elements.append(mpatches.Patch(facecolor=color, alpha=0.25, edgecolor=color, label=tt))

        # Only include vehicle/object categories that actually appear in the data.
        classes_seen = {obj["name"] for obj in object_trajectories.values()}
        legend_elements.append(plt.Line2D([0], [0], color="white", linewidth=0, label="── Vehicles & Objects ──"))
        legend_elements.append(mpatches.Patch(facecolor="green", alpha=0.5, label="Ego Vehicle"))
        if "vehicle" in classes_seen:        legend_elements.append(mpatches.Patch(facecolor="#0F29E9", alpha=0.4, label="Vehicle"))
        if "pedestrian" in classes_seen:     legend_elements.append(mpatches.Patch(facecolor="#DAE93E", alpha=0.4, label="Pedestrian"))
        if "bicycle" in classes_seen:        legend_elements.append(mpatches.Patch(facecolor="#741BBE", alpha=0.4, label="Bicycle"))
        if "traffic_cone" in classes_seen:   legend_elements.append(mpatches.Patch(facecolor="#FFD700", alpha=0.4, label="Traffic Cone"))
        if "generic_object" in classes_seen: legend_elements.append(mpatches.Patch(facecolor="#888888", alpha=0.4, label="Generic Object"))

        ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
                  facecolor="white", labelcolor="black", framealpha=0.9, edgecolor="black")

        # Filename uses the clip-internal position within the 14-frame clip (4..13),
        # matching the multi_view_images naming so BEV / multiview at same index align in time.
        output_path = os.path.join(output_dir, f"{current_i:05d}.jpg")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, format="jpg", bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


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
    parser.add_argument("--maps-dir", default=str(SD_ROOT / "maps"), help="nuPlan maps root (contains nuplan-maps-v1.0.json and one folder per city)")
    parser.add_argument("--scene-filter", default=str(SD_ROOT / "styletest.yaml"),
                        help="StyleDrive scene filter (navsim/planning/script/config/common/train_test_split/scene_filter/styletest.yaml)")
    parser.add_argument("--clips-dir", default=str(SD_ROOT / "StyleClips"))
    args = parser.parse_args()
    pkl_dir = args.logs_dir
    nuplan_maps_root = args.maps_dir
    output_base_path = os.path.join(args.clips_dir, "BEV_images")

    # nuPlan map_api needs this env before importing nuplan.
    os.environ.setdefault("NUPLAN_MAPS_ROOT", nuplan_maps_root)
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    cfg = load_scene_filter(args.scene_filter)
    log_names = cfg["log_names"]
    tokens = cfg["tokens"]
    num_history = cfg["num_history_frames"]
    num_future = cfg["num_future_frames"]
    clip_len = num_history + num_future
    bev_len = CLIP_END_POS - CLIP_START_POS + 1
    print(f"yaml: {len(log_names)} log_names, {len(tokens)} tokens, clip_len={clip_len}, bev_len={bev_len}")

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
    print(f"Will generate {total_clips} clips x {bev_len} BEV frames = {total_clips * bev_len} images")

    # Cache map_api by map_location (loading a nuPlan map takes several seconds).
    map_cache = {}

    def _get_map(name):
        if name not in map_cache:
            print(f"Loading map: {name}")
            map_cache[name] = get_maps_api(nuplan_maps_root, "nuplan-maps-v1.0", name)
        return map_cache[name]

    pbar = tqdm(total=total_clips, desc="Clips")
    for log_name, log_tokens in tokens_by_log.items():
        with open(os.path.join(pkl_dir, f"{log_name}.pkl"), "rb") as f:
            scene_dict_list = pickle.load(f)

        for rep_token in log_tokens:
            out_dir = os.path.join(output_base_path, rep_token)

            # Skip if this clip already has a full set of BEV jpgs.
            if os.path.isdir(out_dir):
                existing = [n for n in os.listdir(out_dir) if n.endswith(".jpg")]
                if len(existing) >= bev_len:
                    pbar.update(1)
                    continue

            try:
                start, end = find_clip(scene_dict_list, rep_token, num_history, num_future)
                clip_pkl_indices = list(range(start, end))
                bev_pkl_indices = clip_pkl_indices[CLIP_START_POS : CLIP_END_POS + 1]
                ref_pkl_idx = clip_pkl_indices[REF_POS]
            except (ValueError, IndexError) as e:
                print(f"[SKIP] {rep_token} ({log_name}): {e}")
                pbar.update(1)
                continue

            ego_positions, ego_yaws, obj_traj, ref_xy_yaw = collect_clip_state(
                scene_dict_list, bev_pkl_indices, ref_pkl_idx
            )

            map_name = scene_dict_list[ref_pkl_idx]["map_location"]
            map_api = _get_map(map_name)
            lanes, triggers = fetch_map_data(map_api, scene_dict_list[ref_pkl_idx], ref_xy_yaw)

            draw_per_frame_bev(ego_positions, ego_yaws, obj_traj, lanes, triggers, out_dir)
            pbar.update(1)
    pbar.close()
    print(f"Done. BEV images saved under: {output_base_path}")


if __name__ == "__main__":
    main()