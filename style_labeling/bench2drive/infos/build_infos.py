#!/usr/bin/env python3
"""
Build the Bench2Drive infos pkls for the labeled scenes:
    <out-dir>/b2d_infos_train.pkl   per-frame infos (ego state, sensors, boxes)
    <out-dir>/b2d_map_infos.pkl     lane / trigger-volume maps of the towns those scenes use

The conversion is Bench2DriveZoo's mmcv/datasets/prepare_B2D.py. That file is licensed
CC BY-NC-ND 4.0, so it is not copied into this repo: clone Bench2DriveZoo and pass --zoo-dir.
This script imports prepare_B2D.py unmodified, points its path constants at our data layout,
and runs it on the scenes in --scenes-file only (the original runs on every Bench2Drive route).

Usage:
    git clone -b uniad/vad https://github.com/Thinklab-SJTU/Bench2DriveZoo.git third_party/Bench2DriveZoo
    python -m style_labeling.bench2drive.infos.build_infos --workers 16
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

from style_labeling.common.paths import B2D_DEFAULT_SCENES_FILE, B2D_ROOT, REPO_ROOT, load_scenes_file


def import_prepare_b2d(zoo_dir: Path):
    datasets_dir = zoo_dir / "mmcv" / "datasets"
    if not (datasets_dir / "prepare_B2D.py").is_file():
        sys.exit("prepare_B2D.py not found under {} (check --zoo-dir)".format(datasets_dir))

    # vis_utils.py imports open3d at module level, but nothing prepare_B2D.py calls uses it
    try:
        import open3d  # noqa: F401
    except ImportError:
        sys.modules["open3d"] = types.ModuleType("open3d")
    # vis_utils.py calls matplotlib.cm.get_cmap, which matplotlib 3.9 removed
    import matplotlib
    import matplotlib.cm
    if not hasattr(matplotlib.cm, "get_cmap"):
        matplotlib.cm.get_cmap = lambda name: matplotlib.colormaps[name]

    sys.path.insert(0, str(datasets_dir))
    return importlib.import_module("prepare_B2D")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zoo-dir", default=str(REPO_ROOT / "third_party" / "Bench2DriveZoo"),
                        help="Bench2DriveZoo checkout (branch uniad/vad)")
    parser.add_argument("--data-root", default=str(B2D_ROOT / "raw"),
                        help="Bench2Drive root containing v1/<Scene>/")
    parser.add_argument("--map-root", default=str(B2D_ROOT / "maps"),
                        help="directory with Town*_HD_map.npz")
    parser.add_argument("--out-dir", default=str(B2D_ROOT / "infos"))
    parser.add_argument("--scenes-file", default=str(B2D_DEFAULT_SCENES_FILE))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-map", action="store_true", help="only build b2d_infos_train.pkl")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    map_root = Path(args.map_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    scenes = load_scenes_file(args.scenes_file)

    missing = [s for s in scenes if not (data_root / "v1" / s / "anno").is_dir()]
    if missing:
        sys.exit("{} scene(s) missing under {}/v1, e.g. {}".format(len(missing), data_root, missing[0]))
    # the town part of the scene name, same as prepare_B2D.py's town_name
    towns = sorted({s.split("_")[1] for s in scenes})
    if not args.skip_map:
        missing = [t for t in towns if not (map_root / "{}_HD_map.npz".format(t)).is_file()]
        if missing:
            sys.exit("map(s) missing under {}: {}".format(map_root, ", ".join(missing)))

    zoo = import_prepare_b2d(Path(args.zoo_dir).resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    zoo.DATAROOT = str(data_root)
    zoo.OUT_DIR = str(out_dir)
    zoo.process_list = []  # a global that prepare_B2D.py's __main__ block normally creates

    workers = max(1, min(args.workers, len(scenes)))
    print("processing {} scenes with {} workers...".format(len(scenes), workers))
    zoo.generate_infos(["v1/" + s for s in scenes], workers, "train", "tmp_data")
    shutil.rmtree(out_dir / "tmp_data")

    if not args.skip_map:
        # gengrate_map() converts every .npz in a directory; give it only the towns we need
        print("processing maps for {}...".format(", ".join(towns)))
        with tempfile.TemporaryDirectory() as tmp:
            for t in towns:
                name = "{}_HD_map.npz".format(t)
                os.symlink(map_root / name, os.path.join(tmp, name))
            zoo.gengrate_map(tmp)
    print("wrote {}".format(out_dir))


if __name__ == "__main__":
    main()
