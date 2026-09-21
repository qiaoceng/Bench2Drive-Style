# ==============================================================================
# [Bench2Drive-Style] NEW FILE -- training-pickle generation
# Author: YinXuan1223   (carried over from the ORION_NYCUACM_version fork)
#
# Generate the planning anchors the diffusion planner denoises from.
# Collects every valid 6-step ego future trajectory out of b2d_infos_train.pkl,
# runs k-means with 20 clusters and writes the (20, 6, 2) anchor array to
# data/others/b2d_plan_anchor_20mode.pkl -- the file the configs reference as
# 'plan_anchor_path'. Run it once, after the info pkl exists and before training
# any use_diff_decoder=True config.
# ==============================================================================
import pickle
import numpy as np
from sklearn.cluster import KMeans

SAMPLE_RATE = 5
PAST_FRAMES = 2
FUTURE_FRAMES = 6
NUM_MODES = 20
OUTPUT_PATH = "data/others/b2d_plan_anchor_20mode.pkl"

print("Loading training data...")
with open("data/infos/b2d_infos_train.pkl", "rb") as f:
    data = pickle.load(f)
print(f"Total training samples: {len(data)}")

all_fut_trajs = []
skipped = 0

for idx in range(len(data)):
    cur_frame = data[idx]
    adj_idx_list = range(
        idx - PAST_FRAMES * SAMPLE_RATE,
        idx + (FUTURE_FRAMES + 1) * SAMPLE_RATE,
        SAMPLE_RATE,
    )
    full_adj_track = np.zeros((PAST_FRAMES + FUTURE_FRAMES + 1, 2))
    full_adj_mask = np.zeros(PAST_FRAMES + FUTURE_FRAMES + 1)
    world2lidar_cur = cur_frame["sensors"]["LIDAR_TOP"]["world2lidar"]

    valid = True
    for j, adj_idx in enumerate(adj_idx_list):
        if adj_idx < 0 or adj_idx >= len(data):
            valid = False
            break
        adj_frame = data[adj_idx]
        if adj_frame["folder"] != cur_frame["folder"]:
            valid = False
            break
        world2lidar_adj = adj_frame["sensors"]["LIDAR_TOP"]["world2lidar"]
        adj2cur = world2lidar_cur @ np.linalg.inv(world2lidar_adj)
        full_adj_track[j, :2] = adj2cur[0:2, 3]
        full_adj_mask[j] = 1

    if not valid or full_adj_mask[-FUTURE_FRAMES:].sum() < FUTURE_FRAMES:
        skipped += 1
        continue

    offset_track = full_adj_track[1:] - full_adj_track[:-1]
    fut_traj = offset_track[PAST_FRAMES:]  # (6, 2)

    # Convert to cumulative trajectory
    cum_traj = np.cumsum(fut_traj, axis=0)
    all_fut_trajs.append(cum_traj)

    if idx % 10000 == 0:
        print(f"  Processed {idx}/{len(data)}, collected {len(all_fut_trajs)} valid trajectories")

all_fut_trajs = np.array(all_fut_trajs)  # (N, 6, 2)
print(f"\nValid trajectories: {all_fut_trajs.shape[0]}, skipped: {skipped}")
print(f"Trajectory shape: {all_fut_trajs.shape}")

# Flatten for K-means: (N, 12)
flat_trajs = all_fut_trajs.reshape(all_fut_trajs.shape[0], -1)
print(f"\nRunning K-means with {NUM_MODES} clusters...")
kmeans = KMeans(n_clusters=NUM_MODES, random_state=42, n_init=10, max_iter=300)
kmeans.fit(flat_trajs)

anchors = kmeans.cluster_centers_.reshape(NUM_MODES, FUTURE_FRAMES, 2)  # (20, 6, 2)
print(f"Anchor shape: {anchors.shape}")

# Sort anchors by final x position for consistency
sort_idx = np.argsort(anchors[:, -1, 0])
anchors = anchors[sort_idx]

print("\nAnchor final positions (x, y) at t=6:")
for i, a in enumerate(anchors):
    print(f"  Mode {i:2d}: ({a[-1, 0]:7.2f}, {a[-1, 1]:7.2f})")

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(anchors, f)
print(f"\nSaved anchors to {OUTPUT_PATH}")
