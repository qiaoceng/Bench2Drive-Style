"""
StyleDrive driving-style inference with Qwen3-VL (vLLM), used to validate the labeler
against StyleDrive's ground-truth style labels. Resumable per clip token; can be split
across GPUs/processes with --chunk/--num-chunks.

Each StyleDrive clip token = one 5-s clip. Inputs: the classification prompt, the ego-status text,
10 multi-view composites (clip positions 4..13) and 10 BEV frames.

Output: <results-dir>/<run-tag>/<clip_token>/check_response.json

Usage:
    # all 4049 clips on two GPUs (tensor parallel), as in the reported results
    CUDA_VISIBLE_DEVICES=0,1 python -m style_labeling.styledrive.inference.run_vlm --run-tag v17_sd_all

    # random subset, split into two single-GPU processes writing to the same run dir
    CUDA_VISIBLE_DEVICES=0 python -m style_labeling.styledrive.inference.run_vlm --n 300 --tp 1 --run-tag v17_sd_300 --chunk 0 --num-chunks 2
    CUDA_VISIBLE_DEVICES=1 python -m style_labeling.styledrive.inference.run_vlm --n 300 --tp 1 --run-tag v17_sd_300 --chunk 1 --num-chunks 2
"""
import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from ...common.parsing import parse_label
from ...common.paths import SD_ROOT
from ...common.vlm import (CHECK_RESPONSE_NAME, add_vlm_args, ask_vlm, build_llm,
                          build_messages, read_text)


def clip_image_paths(clips_dir: Path, clip_token: str) -> list:
    """10 multi-view frames (clip positions 4..13, the future part) followed by the 10 BEV frames."""
    mv_dir = clips_dir / "multi_view_images" / clip_token
    mv_paths = [str(mv_dir / f"{k:05d}.jpg") for k in range(4, 14)]
    bev_dir = clips_dir / "BEV_images" / clip_token
    bev_paths = sorted(str(f) for f in bev_dir.iterdir() if f.is_file())
    return mv_paths + bev_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips-dir", default=str(SD_ROOT / "StyleClips"),
                        help="Output of the prepare_* scripts (multi_view_images/, BEV_images/, ego_status_in_text_enhanced/).")
    parser.add_argument("--ego-status-json", default=None,
                        help="Default: <clips-dir>/ego_status_in_text_enhanced/ego_status_in_text_enhanced.json")
    parser.add_argument("--results-dir", default=str(SD_ROOT / "StyleResults"))
    parser.add_argument("--n", type=int, default=None, help="Random subset size (default: all clips).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--run-tag", default=None,
                        help="Run directory name to create or resume. Give the same tag to every chunk. "
                             "Default: <timestamp>_v17_sd_<n|all>.")
    add_vlm_args(parser)
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    ego_status_json = args.ego_status_json or clips_dir / "ego_status_in_text_enhanced" / "ego_status_in_text_enhanced.json"
    system_prompt = read_text(args.system_prompt_path)
    classification_prompt = read_text(args.prompt_path)

    tag = args.run_tag or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_v17_sd_{args.n or 'all'}"
    run_dir = Path(args.results_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Chunk {args.chunk}/{args.num_chunks}  Run: {run_dir}", flush=True)

    items = json.load(open(ego_status_json, encoding="utf-8"))
    print(f"  {len(items)} total clips  (ego_status={ego_status_json})", flush=True)
    if args.n:
        random.seed(args.seed)
        items = random.sample(items, args.n)
    chunk = [s for i, s in enumerate(items) if i % args.num_chunks == args.chunk]
    print(f"  chunk {len(chunk)} clips (seed={args.seed})", flush=True)

    llm, sp = build_llm(args, [clips_dir])

    t0 = time.time()
    cnt = {"aggressive": 0, "normal": 0, "conservative": 0, "unknown": 0}
    for i, item in enumerate(chunk):
        tok = item["clip_token"]
        out_path = run_dir / tok / CHECK_RESPONSE_NAME
        if out_path.exists():
            print(f"[C{args.chunk}:{i+1}/{len(chunk)}] {tok} SKIP", flush=True)
            continue
        try:
            t = time.time()
            messages = build_messages(system_prompt, classification_prompt, item["content"][1]["text"],
                                      clip_image_paths(clips_dir, tok))
            resp = ask_vlm(llm, sp, messages, args.model)
            dt = time.time() - t
            lab, _ = parse_label(resp["choices"][0]["message"]["content"])
            cnt[lab if lab in cnt else "unknown"] += 1
            (run_dir / tok).mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([{"scene_name": tok, "start_idx": 0, "response": resp}], f, ensure_ascii=False, indent=2)
            print(f"[C{args.chunk}:{i+1}/{len(chunk)}] {tok} label={lab:<13s} "
                  f"comp={resp['usage']['completion_tokens']} {dt:.1f}s  "
                  f"agg:{cnt['aggressive']} nor:{cnt['normal']} con:{cnt['conservative']} unk:{cnt['unknown']}", flush=True)
        except Exception as e:
            print(f"[C{args.chunk}:{i+1}/{len(chunk)}] {tok} ERROR: {e}", flush=True)
    print(f"\n[C{args.chunk}] Done {(time.time()-t0)/60:.1f}min  counts={cnt}", flush=True)


if __name__ == "__main__":
    main()
