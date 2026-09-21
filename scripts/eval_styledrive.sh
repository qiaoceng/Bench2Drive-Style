#!/usr/bin/env bash
# StyleDrive: build VLM inputs, label all 4049 test clips, and compare with the ground-truth style labels.
# Run from the repo root. Assumes the layout in docs/DataPreparation.md.
set -euo pipefail

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_v17_sd_all}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}   # 2 GPUs (tensor parallel)

python -m style_labeling.styledrive.prepare.prepare_multiview
python -m style_labeling.styledrive.prepare.prepare_bev
python -m style_labeling.styledrive.prepare.prepare_ego_context

python -m style_labeling.styledrive.inference.run_vlm --run-tag "$RUN_TAG"
python -m style_labeling.styledrive.evaluation.evaluate "data/StyleDrive/StyleResults/$RUN_TAG"
