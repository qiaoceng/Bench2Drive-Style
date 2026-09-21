#!/usr/bin/env bash
# ==============================================================================
# [Bench2Drive-Style] NEW FILE -- script
# Author: YinXuan1223   (carried over from the ORION_NYCUACM_version fork)
#
# Launcher for the VAE+style fine-tune resume: single GPU, bf16,
# torch.distributed.launch with nproc_per_node=1, resuming from iter_4000 for one
# full 16,896-frame pass (max_iters=20896), logging to <work_dir>/logs/train.<date>.
# The repo root is derived from the script's own location. CFG and WORK_DIR are still
# absolute paths from the original machine -- set those two before running.
# Kept as the record of the original run; to train in this repo use
# scripts/train_vae_style.sh (same launch, paths and resume set for this repo).
# ==============================================================================
# [VAE+Style RESUME] Resume the frozen VAE+style fine-tune from iter_4000 for one full
# 16,896-frame pass (-> max_iters=20896). Single GPU 5, bf16, checkpoints to HDD6/yinxuan.
set -o pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

# [Bench2Drive-Style] repo root derived from this script's own location
# (adzoo/orion/ -> ../..), replacing the old hard-coded 'Ourion' path.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"   # cwd = repo root so data/ and ckpts/ resolve

CFG=/mnt/HDD8/dow904/orion_stage2_vae_style_ft_resume.py
WORK_DIR=/mnt/HDD6/yinxuan/dow904_ckpt
mkdir -p ${WORK_DIR}/logs
T=$(date +%m%d%H%M)

PYTHONPATH="$REPO":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=54643 \
    --nnodes=1 --node_rank=0 \
    adzoo/orion/train.py ${CFG} \
    --launcher pytorch --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}/logs/train.${T}
