"""
Parse a StyleDrive run and compare the VLM labels with StyleDrive's ground truth.

Writes into the run dir:
    parsed_reasons_vllm.json   one parsed record per clip token
    comparison_report.txt      confusion matrix, per-class P/R/F1, accuracy, macro F1,
                               distribution, accuracy by scenario_type and speed_mode
    confusion_matrix.png
    label_distribution.png

Usage:
    python -m style_labeling.styledrive.evaluation.evaluate data/StyleDrive/StyleResults/<run-tag>
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...common.parsing import parse_response
from ...common.paths import SD_ROOT
from ...common.vlm import CHECK_RESPONSE_NAME

CLASSES = ["conservative", "normal", "aggressive"]
PRED_COLS = CLASSES + ["unknown"]
GT_MAP = {"C": "conservative", "N": "normal", "A": "aggressive"}
STYLE_COLORS = {"conservative": "#4C78A8", "normal": "#72B7B2",
                "aggressive": "#E45756", "unknown": "#999999"}


def parse_all(run_dir: Path):
    scene_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / CHECK_RESPONSE_NAME).exists())
    all_data = []
    for sd in scene_dirs:
        with open(sd / CHECK_RESPONSE_NAME, encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            content_str = item["response"]["choices"][0]["message"]["content"]
            all_data.append({
                "scene_name": item.get("scene_name", sd.name),
                "index": item.get("start_idx", 0),
                **parse_response(content_str),
                "usage": item["response"].get("usage", {}),
            })
    return all_data


def compare(data, gt):
    """Return (report text, confusion matrix [GT x PRED_COLS])."""
    cm = np.zeros((len(CLASSES), len(PRED_COLS)), dtype=int)
    by_group = {"scenario_type": defaultdict(lambda: [0, 0, 0]), "speed_mode": defaultdict(lambda: [0, 0, 0])}
    missing_gt = 0
    for d in data:
        g = gt.get(d["scene_name"])
        if g is None or g.get("ANC_result") not in GT_MAP:
            missing_gt += 1
            continue
        gl = GT_MAP[g["ANC_result"]]
        pl = d["label"] if d["label"] in CLASSES else "unknown"
        cm[CLASSES.index(gl), PRED_COLS.index(pl)] += 1
        for key, groups in by_group.items():
            s = groups[g.get(key, "?")]
            s[0] += 1                      # n
            s[1] += int(pl == gl)          # correct
            s[2] += int(pl == "unknown")   # unknown
    n = int(cm.sum())

    L = ["Confusion Matrix (rows = GT, cols = pred)", ""]
    L.append(" " * 14 + "".join(f"{c[:5]:>9s}" for c in PRED_COLS))
    for i, c in enumerate(CLASSES):
        L.append(f"  {c:>12s}" + "".join(f"{v:9d}" for v in cm[i]) + f"  | total={cm[i].sum():5d}")
    L.append(f"  {'pred_total':>12s}" + "".join(f"{v:9d}" for v in cm.sum(0)))

    L += ["", "Per-class metrics (excluding 'unknown' preds for precision/recall)",
          f"  {'class':<15s}{'precision':>10s}{'recall':>9s}{'f1':>7s}{'support_gt':>13s}{'support_pred':>14s}"]
    f1s = []
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        pred = cm[:, i].sum()
        sup = cm[i].sum()
        sup_known = sup - cm[i, -1]  # GT clips whose prediction is not 'unknown'
        p = tp / pred if pred else 0.0
        r = tp / sup_known if sup_known else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        L.append(f"  {c:<15s}{p:10.3f}{r:9.3f}{f1:7.3f}{sup:13d}{pred:14d}")

    unk = cm[:, -1].sum()
    correct = int(np.trace(cm[:, :len(CLASSES)]))
    L += ["", f"Overall accuracy (excl unknown): {correct}/{n - unk} = {correct / max(n - unk, 1):.3f}",
          f"Macro F1: {np.mean(f1s):.3f}",
          f"Predictions labelled 'unknown' : {unk}/{n} = {unk / max(n, 1):.3f}"]
    if missing_gt:
        L.append(f"Clips without GT (skipped)     : {missing_gt}")

    L += ["", "Distribution comparison",
          f"  {'class':<15s}{'gt_count':>9s}{'gt_pct':>8s}{'pred_count':>12s}{'pred_pct':>10s}"]
    for j, c in enumerate(PRED_COLS):
        gc = cm[j].sum() if c in CLASSES else 0
        pc = cm[:, j].sum()
        L.append(f"  {c:<15s}{gc:9d}{100 * gc / n:7.2f}%{pc:12d}{100 * pc / n:9.2f}%")

    for key, groups in by_group.items():
        L += ["", f"Accuracy by {key}", f"  {key:<30s}{'n':>5s}{'correct':>9s}{'acc':>7s}{'unk':>6s}"]
        for name, (cnt, cor, u) in sorted(groups.items(), key=lambda x: -x[1][0]):
            L.append(f"  {name:<30s}{cnt:5d}{cor:9d}{cor / cnt:7.3f}{u:6d}")
    return "\n".join(L) + "\n", cm


def plot_confusion(cm, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(PRED_COLS)), PRED_COLS)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    for i in range(cm.shape[0]):
        row = cm[i].sum()
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({100 * cm[i, j] / row:.0f}%)" if row else "0", ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"StyleDrive — VLM vs GT (n={cm.sum()})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_label_distribution(data, out_path: Path):
    counts = {lab: 0 for lab in PRED_COLS}
    for d in data:
        counts[d["label"] if d["label"] in counts else "unknown"] += 1
    total = sum(counts.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [counts[l] for l in PRED_COLS]
    bars = ax.bar(PRED_COLS, vals, color=[STYLE_COLORS[l] for l in PRED_COLS], edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        pct = 100 * v / total if total else 0
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Clips", fontsize=9)
    ax.set_title(f"StyleDrive — Predicted Label Distribution (n={total})", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.15 if vals else 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="StyleDrive run dir (contains per-clip_token subfolders)")
    parser.add_argument("--gt-json", default=str(SD_ROOT / "styletest.json"),
                        help="StyleDrive ground truth (dataset/extra_data/styletest.json); label field = ANC_result")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    print(f"Parsing {run_dir} ...")
    data = parse_all(run_dir)
    print(f"  parsed {len(data)} clips")
    with open(run_dir / "parsed_reasons_vllm.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open(args.gt_json, encoding="utf-8") as f:
        gt = json.load(f)
    report, cm = compare(data, gt)
    (run_dir / "comparison_report.txt").write_text(report, encoding="utf-8")
    plot_confusion(cm, run_dir / "confusion_matrix.png")
    plot_label_distribution(data, run_dir / "label_distribution.png")
    print(report)
    print(f"Wrote parsed_reasons_vllm.json, comparison_report.txt, confusion_matrix.png, label_distribution.png to {run_dir}")


if __name__ == "__main__":
    main()
