#!/usr/bin/env bash
# StyleDrive: download the test-split data used for evaluation into data/StyleDrive/:
#   styletest.json + styletest.yaml, OpenScene v1.1 test metadata and camera images (no LiDAR), nuPlan maps.
# Run from the repo root. Needs the Hugging Face CLI (pip install -U huggingface_hub), curl and unzip.
# About 130 GB is downloaded; archives are unpacked and deleted one at a time. Re-run to resume.
set -euo pipefail

SD=data/StyleDrive
TMP=$SD/.download
OPENSCENE=OpenDriveLab/OpenScene
STYLEDRIVE_COMMIT=4b9392c92cd2c253fedec01bfdf4a412a8bcd4f1
mkdir -p "$SD" "$TMP"

# 1. StyleDrive ground truth and scene filter (4049 test clips)
[ -f "$SD/styletest.json" ] || hf download Ryhn98/StyleDrive-Dataset styletest.json --repo-type dataset --local-dir "$SD"
[ -f "$SD/styletest.yaml" ] || curl -fL --progress-bar -o "$SD/styletest.yaml" \
    "https://raw.githubusercontent.com/AIR-THU/StyleDrive/$STYLEDRIVE_COMMIT/navsim/planning/script/config/common/train_test_split/scene_filter/styletest.yaml"

# 2. OpenScene test metadata -> navsim_logs/test/
if [ ! -f "$TMP/metadata.done" ]; then
    f=openscene-v1.1/openscene_metadata_test.tgz
    hf download "$OPENSCENE" "$f" --repo-type dataset --local-dir "$TMP"
    tar xzf "$TMP/$f" -C "$SD" --transform 's#^openscene-v1\.1/meta_datas#navsim_logs#'
    rm "$TMP/$f"
    touch "$TMP/metadata.done"
fi

# 3. OpenScene test camera images (32 parts, ~4 GB each) -> sensor_blobs/test/
for i in $(seq 0 31); do
    [ -f "$TMP/camera_$i.done" ] && continue
    f=openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_$i.tgz
    hf download "$OPENSCENE" "$f" --repo-type dataset --local-dir "$TMP"
    tar xzf "$TMP/$f" -C "$SD" --transform 's#^openscene-v1\.1/##'
    rm "$TMP/$f"
    touch "$TMP/camera_$i.done"
done

# 4. nuPlan maps -> maps/ (nuplan-maps-v1.0.json + one folder per city)
if [ ! -f "$SD/maps/nuplan-maps-v1.0.json" ]; then
    curl -fL --progress-bar -o "$TMP/nuplan-maps-v1.1.zip" \
        https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
    unzip -q "$TMP/nuplan-maps-v1.1.zip" -d "$TMP"
    mv "$TMP/nuplan-maps-v1.0" "$SD/maps"
    rm "$TMP/nuplan-maps-v1.1.zip"
fi

rm -rf "$TMP"
echo "done: $(ls "$SD/navsim_logs/test" | wc -l) logs in $SD/navsim_logs/test, sensor_blobs/test, maps, styletest.{json,yaml}"
