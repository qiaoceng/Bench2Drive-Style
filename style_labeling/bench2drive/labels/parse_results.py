"""
Parse a Bench2Drive run: check_response.json → parsed_reasons_vllm.json per scene, plus two plots.

parsed_reasons_vllm.json is the input of build_style_pkl.py.

Usage:
    python -m style_labeling.bench2drive.labels.parse_results data/Bench2Drive/StyleResults/<run-tag>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...common.parsing import parse_response
from ...common.vlm import CHECK_RESPONSE_NAME


def plot_gaussian(parsed_json_path, sigma: float, output_dir):
    """Clip-level style score (aggressive=1, normal=0, conservative=-1) after Gaussian smoothing."""
    with open(parsed_json_path) as f:
        data = json.load(f)
    label_map = {"aggressive": 1.0, "normal": 0.0, "conservative": -1.0}
    pts = sorted(((d["index"], label_map[d["label"]]) for d in data if d["label"] in label_map),
                 key=lambda x: x[0])
    if not pts:
        print(f"  [skip gaussian] no labelled points in {Path(parsed_json_path).parent.name}")
        return
    indices = np.array([p[0] for p in pts])
    scores  = np.array([p[1] for p in pts], dtype=np.float64)
    smooth  = gaussian_filter1d(scores, sigma=sigma)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.fill_between(indices, smooth, 0.33, where=smooth >= 0.33, color="#E45756", alpha=0.4)
    ax.fill_between(indices, smooth, -0.33, where=smooth <= -0.33, color="#4C78A8", alpha=0.4)
    ax.plot(indices, smooth, color="black", linewidth=1)
    ax.axhline(0.33,  color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axhline(-0.33, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([-1, -0.33, 0, 0.33, 1])
    ax.set_yticklabels(["Conservative", "", "Normal", "", "Aggressive"], fontsize=7)
    ax.set_title(f"Gaussian-Smoothed Style (σ={sigma}, n={len(indices)})", fontsize=9)
    ax.set_xlabel("Clip Index", fontsize=9)
    plt.tight_layout()
    out = Path(output_dir) / "style_gaussian.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved   → {out.name}")


def parse_scene(scene_dir: Path):
    raw_path = scene_dir / CHECK_RESPONSE_NAME
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    label_counts = {"aggressive": 0, "normal": 0, "conservative": 0, "unknown": 0}
    all_data = []
    for item in raw_items:
        content_str = item["response"]["choices"][0]["message"]["content"]
        parsed = parse_response(content_str)
        label_counts[parsed["label"] if parsed["label"] in label_counts else "unknown"] += 1
        all_data.append({
            "scene_name": item.get("scene_name", scene_dir.name),
            "index": item.get("start_idx"),
            **parsed,
            "usage": item["response"].get("usage", {}),
        })

    all_data.sort(key=lambda x: x.get("index", -1) if x.get("index") is not None else -1)
    parsed_path = scene_dir / "parsed_reasons_vllm.json"
    with open(parsed_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    return all_data, label_counts, parsed_path


# ── style_dist_nconf plot ─────────────────────────────────────────────────────
STYLE_COLORS = {"conservative": "#4C78A8", "normal": "#72B7B2", "aggressive": "#E45756"}


def plot_style_nconf(all_data, label_counts, scene_dir: Path, prompt_version: str = "v17", temp: str = "0.6"):
    abbr = scene_dir.name[:24]
    records = [(d["index"], d["label"]) for d in all_data if d["label"] in STYLE_COLORS]
    if not records:
        print(f"  [skip nconf] {scene_dir.name}: no styled records")
        return None
    records.sort()

    fig, ax = plt.subplots(figsize=(16, 2.5))
    ax.bar(
        [r[0] for r in records], [1] * len(records),
        color=[STYLE_COLORS[r[1]] for r in records], width=0.8, linewidth=0,
    )
    ax.set_xlim(-0.5, max(r[0] for r in records) + 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Clip Index")
    ax.set_title(f"Style Distribution — {abbr} ({prompt_version}, vllm, t={temp}, n={len(records)})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=STYLE_COLORS[s], label=s)
               for s in ("conservative", "normal", "aggressive")]
    ax.legend(handles=handles, title="Style", loc="upper right",
              fontsize=7, title_fontsize=7, handlelength=1, borderpad=0.3)
    plt.tight_layout()
    save_path = scene_dir / "style_dist_nconf.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Run dir (contains scene subfolders with check_response.json)")
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="Smoothing for style_gaussian.png (4 = the value used by build_style_pkl.py)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    scene_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / CHECK_RESPONSE_NAME).exists())
    print(f"Found {len(scene_dirs)} scene(s) in {run_dir}")

    for scene_dir in scene_dirs:
        print(f"\n[{scene_dir.name}]")
        all_data, label_counts, parsed_path = parse_scene(scene_dir)
        total = sum(label_counts.values())
        print(f"  parsed → {parsed_path.name}  "
              f"agg={label_counts['aggressive']} nor={label_counts['normal']} "
              f"con={label_counts['conservative']} unk={label_counts['unknown']} total={total}")

        nconf_path = plot_style_nconf(all_data, label_counts, scene_dir)
        if nconf_path:
            print(f"  saved   → {nconf_path.name}")
        plot_gaussian(str(parsed_path), sigma=args.sigma, output_dir=str(scene_dir))


if __name__ == "__main__":
    main()
