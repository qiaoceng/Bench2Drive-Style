# Data Preparation

For fine-tuning the planner on the labeled data, see [TrainingVAEStyle.md](TrainingVAEStyle.md).

Nothing under `data/` is tracked by git. Put the datasets here, or symlink them in. Every script also takes flags (`--clips-dir`, `--infos-pkl`, `--logs-dir`, ...) that point somewhere else instead.

## Bench2Drive

Only the 36 scenes in `style_labeling/bench2drive/splits/labeled_scenes.json` are needed. They come from Bench2Drive **Base** ([GitHub](https://github.com/Thinklab-SJTU/Bench2Drive), [Hugging Face](https://huggingface.co/datasets/rethinklab/Bench2Drive)). The HD maps come from [rethinklab/Bench2Drive-Map](https://huggingface.co/datasets/rethinklab/Bench2Drive-Map).

### 1. Download

```bash
pip install -U huggingface_hub          # provides the `hf` command
bash scripts/download_bench2drive.sh
```

The script reads the scene list and then does the following:
- downloads the 36 scene archives (about 11 GB) to `data/Bench2Drive/archives/` and unpacks them to `data/Bench2Drive/raw/v1/<Scene>/`;
- downloads the HD maps of the 7 towns those scenes use (Town03/04/05/06/12/13/15, about 3.5 GB) to `data/Bench2Drive/maps/`.

You can delete `archives/` once the scenes are unpacked. The script is equivalent to:

```bash
hf download rethinklab/Bench2Drive <Scene>.tar.gz ... --repo-type dataset --local-dir data/Bench2Drive/archives
hf download rethinklab/Bench2Drive-Map Town03_HD_map.npz ... --repo-type dataset --local-dir data/Bench2Drive/maps
tar xzf data/Bench2Drive/archives/<Scene>.tar.gz -C data/Bench2Drive/raw/v1
```

To use a different scene list, run `SCENES_FILE=<list.json> bash scripts/download_bench2drive.sh`.

### 2. Build the infos pkls

The infos are built with `prepare_B2D.py` from [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo). That file is licensed CC BY-NC-ND 4.0, so it is not included here. Clone Bench2DriveZoo into `third_party/`. Our wrapper imports the file unmodified and runs it on the labeled scenes only:

```bash
git clone -b uniad/vad https://github.com/Thinklab-SJTU/Bench2DriveZoo.git third_party/Bench2DriveZoo
pip install -r style_labeling/requirements.txt   # includes opencv-python, pyquaternion
python -m style_labeling.bench2drive.infos.build_infos --workers 16
```

This writes two files to `data/Bench2Drive/infos/`:
- `b2d_infos_train.pkl` (~80 MB): per-frame infos (ego state, camera/LiDAR calibration, 3D boxes) for the 36 scenes;
- `b2d_map_infos.pkl` (~4.4 GB): lanes and trigger volumes of the 7 towns.

Frames without any visible box are dropped by `prepare_B2D.py`, so the 36 scenes' 8389 frames give 8384 entries. Use `--zoo-dir` if Bench2DriveZoo is somewhere else, and `--skip-map` to rebuild only the frame infos.

## StyleDrive

[StyleDrive](https://github.com/AIR-THU/StyleDrive) adds driving-style labels to the NAVSIM / OpenScene data, which is built on nuPlan. We only use its **test split**: 4049 clips from 136 logs. The labeler needs the following inputs:

| File | Source |
|---|---|
| `styletest.json`: ground-truth style labels (field `ANC_result`) | Hugging Face [Ryhn98/StyleDrive-Dataset](https://huggingface.co/datasets/Ryhn98/StyleDrive-Dataset) |
| `styletest.yaml`: scene filter listing the 4049 test tokens | StyleDrive repo, `navsim/planning/script/config/common/train_test_split/scene_filter/styletest.yaml` |
| `navsim_logs/test/`: OpenScene v1.1 test metadata (147 log pkls) | Hugging Face [OpenDriveLab/OpenScene](https://huggingface.co/datasets/OpenDriveLab/OpenScene), `openscene-v1.1/openscene_metadata_test.tgz` |
| `sensor_blobs/test/`: camera images | the same repo, `openscene-v1.1/openscene_sensor_test_camera/` (32 parts) |
| `maps/`: nuPlan maps (`nuplan-maps-v1.0`) | [nuPlan](https://www.nuscenes.org/nuplan), `nuplan-maps-v1.1.zip` |

Run:

```bash
pip install -U huggingface_hub          # provides the `hf` command
bash scripts/download_styledrive.sh
```

The script downloads about 130 GB, almost all of it camera images. It unpacks and deletes the archives one at a time. If it stops, run it again to resume. It skips the OpenScene LiDAR parts because the labeler only uses the cameras. `styletest.yaml` is fetched from a pinned StyleDrive commit.

Compared with StyleDrive's own [download instructions](https://github.com/AIR-THU/StyleDrive/blob/main/docs/install.md), the script puts `styletest.json` directly under `data/StyleDrive/` instead of `extra_data/`. It also renames `test_navsim_logs` / `test_sensor_blobs` to `navsim_logs/test` / `sensor_blobs/test`.

`prepare_bev.py` also needs the nuPlan map API. Install nuplan-devkit as described in the NAVSIM / StyleDrive setup. The script sets `NUPLAN_MAPS_ROOT` to `data/StyleDrive/maps` unless it is already set.

## Data Layout
```
data/
├── Bench2Drive/
│   ├── archives/                            # downloaded <Scene>.tar.gz; can be deleted after unpacking
│   ├── raw/v1/<Scene>/                      # unpacked scenes (anno/, camera/, expert_assessment/, ...)
│   ├── maps/Town*_HD_map.npz                # HD maps
│   ├── infos/
│   │   ├── b2d_infos_train.pkl              # frame infos of the 36 scenes (build_infos.py)
│   │   ├── b2d_map_infos.pkl                # maps of their towns (build_infos.py)
│   │   └── b2d_infos_train_with_style.pkl   # b2d_infos_train.pkl + driving_style (build_style_pkl.py)
│   ├── StyleClips/                          # written by style_labeling/bench2drive/prepare/
│   │   ├── multi_view_images/v1/<Scene>/<frame#>.jpg
│   │   ├── BEV_images/<Scene>/clip_<i>/*.jpg
│   │   └── ego_status_in_text_enhanced/<Scene>.json
│   └── StyleResults/                        # written by run_vlm.py, one folder per run
│
└── StyleDrive/
    ├── navsim_logs/test/<log>.pkl           # OpenScene v1.1 test metadata
    ├── sensor_blobs/test/<log>/CAM_*/*.jpg  # OpenScene v1.1 test camera images
    ├── maps/                                # nuPlan maps: nuplan-maps-v1.0.json + one folder per city
    ├── styletest.json                       # ground truth (field ANC_result)
    ├── styletest.yaml                       # scene filter of the 4049 test clips
    ├── StyleClips/                          # written by style_labeling/styledrive/prepare/
    │   ├── multi_view_images/<clip_token>/<frame#>.jpg
    │   ├── BEV_images/<clip_token>/<frame#>.jpg
    │   └── ego_status_in_text_enhanced/ego_status_in_text_enhanced.json
    └── StyleResults/
```
