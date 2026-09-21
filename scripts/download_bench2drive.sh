#!/usr/bin/env bash
# Bench2Drive: download the labeled scenes and the HD maps of their towns, and unpack the scenes.
# Run from the repo root. Needs the Hugging Face CLI (pip install -U huggingface_hub).
set -euo pipefail

B2D=data/Bench2Drive
SCENES_FILE=${SCENES_FILE:-style_labeling/bench2drive/splits/labeled_scenes.json}

mapfile -t SCENES < <(python -c "import sys; from style_labeling.common.paths import load_scenes_file; print('\n'.join(load_scenes_file(sys.argv[1])))" "$SCENES_FILE")
mapfile -t TOWNS < <(printf '%s\n' "${SCENES[@]}" | cut -d_ -f2 | sort -u)

hf download rethinklab/Bench2Drive "${SCENES[@]/%/.tar.gz}" --repo-type dataset --local-dir "$B2D/archives"
hf download rethinklab/Bench2Drive-Map "${TOWNS[@]/%/_HD_map.npz}" --repo-type dataset --local-dir "$B2D/maps"

mkdir -p "$B2D/raw/v1"
for s in "${SCENES[@]}"; do
    [ -d "$B2D/raw/v1/$s" ] || tar xzf "$B2D/archives/$s.tar.gz" -C "$B2D/raw/v1"
done
echo "done: ${#SCENES[@]} scenes in $B2D/raw/v1, maps for ${TOWNS[*]} in $B2D/maps"
