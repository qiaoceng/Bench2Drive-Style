#!/usr/bin/env bash
# Bench2Drive: build VLM inputs, label every clip with Qwen3-VL, and write driving_style into the training pkl.
# Run from the repo root. Assumes the layout in docs/DataPreparation.md.
set -euo pipefail

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_v17_b2d}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}   # 2 GPUs (tensor parallel)

python -m style_labeling.bench2drive.prepare.prepare_multiview
python -m style_labeling.bench2drive.prepare.prepare_bev
python -m style_labeling.bench2drive.prepare.prepare_ego_context

python -m style_labeling.bench2drive.inference.run_vlm --run-tag "$RUN_TAG"
python -m style_labeling.bench2drive.labels.parse_results "data/Bench2Drive/StyleResults/$RUN_TAG"

python -m style_labeling.bench2drive.labels.build_style_pkl \
    --style-dir "data/Bench2Drive/StyleResults/$RUN_TAG" \
    --pkl-in data/Bench2Drive/infos/b2d_infos_train.pkl \
    --pkl-out data/Bench2Drive/infos/b2d_infos_train_with_style.pkl \
    --sigma 4
