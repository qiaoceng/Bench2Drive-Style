#!/usr/bin/env bash
# StyleOrion: fine-tune ORION's VAE planner with the per-frame driving_style labels (single GPU, bf16).
# Run from the repo root, after scripts/label_bench2drive.sh. See docs/TrainingVAEStyle.md.
# Same launch as StyleOrion/adzoo/orion/run_vae_style_resume.sh, the launcher of the original run.
#
#   bash scripts/train_vae_style.sh                                          # start from ckpts/Orion.pth
#   RESUME_FROM=<work_dir>/iter_4000.pth bash scripts/train_vae_style.sh     # resume a run
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
REPO=$(pwd)
CFG=${CFG:-adzoo/orion/configs/orion_stage2_vae_style_ft.py}     # relative to StyleOrion/
WORK_DIR=${WORK_DIR:-$REPO/StyleOrion/adzoo/orion/work_dirs/orion_stage2_vae_style_ft}
LOAD_FROM=${LOAD_FROM:-$REPO/StyleOrion/ckpts/Orion.pth}
RESUME_FROM=${RESUME_FROM:-}
MAX_ITERS=${MAX_ITERS:-20896}          # total iterations; a resumed run continues up to this number
MASTER_PORT=${MASTER_PORT:-54643}
INFOS=$REPO/data/Bench2Drive/infos
WORK_DIR=$(realpath -m "$WORK_DIR"); LOAD_FROM=$(realpath -m "$LOAD_FROM")   # we cd into StyleOrion/ below

# The config resolves data/... relative to StyleOrion/ (see docs/TrainingStyleOrion.md, "Data layout").
mkdir -p StyleOrion/data/bench2drive
link() { [ -e "$2" ] || [ -L "$2" ] || ln -s "$1" "$2"; }   # never replace what is already there
link "$REPO/data/Bench2Drive/raw/v1" StyleOrion/data/bench2drive/v1
link "$REPO/data/Bench2Drive/maps"   StyleOrion/data/bench2drive/maps
link "$INFOS"                        StyleOrion/data/infos

# The config reads b2d_infos_train_style_mixed.pkl, which has the same content as
# b2d_infos_train_with_style.pkl written by scripts/label_bench2drive.sh.
[ -e "$INFOS/b2d_infos_train_with_style.pkl" ] && link b2d_infos_train_with_style.pkl "$INFOS/b2d_infos_train_style_mixed.pkl"
for f in "$INFOS/b2d_infos_train_style_mixed.pkl" "$INFOS/b2d_map_infos.pkl" StyleOrion/ckpts/pretrain_qformer; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

# resume_from wins over load_from; train.py's own --resume-from is ignored silently if the file is
# missing, so pass it through --cfg-options and check it here.
OPTS=(load_from="$LOAD_FROM" runner.max_iters="$MAX_ITERS")
if [ -n "$RESUME_FROM" ]; then
    [ -f "$RESUME_FROM" ] || { echo "missing: $RESUME_FROM" >&2; exit 1; }
    OPTS+=(resume_from="$(realpath "$RESUME_FROM")")
else
    [ -f "$LOAD_FROM" ] || { echo "missing: $LOAD_FROM" >&2; exit 1; }
fi

mkdir -p "$WORK_DIR/logs"
T=$(date +%m%d%H%M)
cd StyleOrion   # the config's llm_path / data paths are relative to StyleOrion/

# --no-validate: the config never evaluates (evaluation.interval > max_iters), and without the
# flag train.py still builds the val set from b2d_infos_val.pkl, which this repo does not produce.
PYTHONPATH="$REPO/StyleOrion${PYTHONPATH:+:$PYTHONPATH}" \
python -m torch.distributed.launch \
    --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
    --nnodes=1 --node_rank=0 \
    adzoo/orion/train.py "$CFG" \
    --launcher pytorch --deterministic --no-validate \
    --work-dir "$WORK_DIR" \
    --cfg-options "${OPTS[@]}" \
    2>&1 | tee "$WORK_DIR/logs/train.$T"
