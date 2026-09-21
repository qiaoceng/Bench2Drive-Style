"""
Bench2Drive driving-style inference with Qwen3-VL (vLLM), resumable per scene/clip.

Each clip = 10 frames at 2Hz (5 s). Inputs per clip: the v17 prompt, the ego-status
text, 10 multi-view composites and the clip's BEV frames.

Output: <results-dir>/<run-tag>/<scene>/check_response.json

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python -m style_labeling.bench2drive.inference.run_vlm
    CUDA_VISIBLE_DEVICES=0,1 python -m style_labeling.bench2drive.inference.run_vlm --scene HighwayCutIn
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from ...common.parsing import parse_label
from ...common.paths import B2D_DEFAULT_SCENES_FILE, B2D_ROOT, load_scenes_file
from ...common.vlm import (CHECK_RESPONSE_NAME, add_vlm_args, ask_vlm, build_llm,
                          build_messages, read_text)


def clip_image_paths(clips_dir: Path, scene_name: str, start_idx: int) -> list:
    """10 multi-view frames (every 5th 10Hz frame from start_idx) followed by the clip's BEV frames."""
    mv_dir = clips_dir / "multi_view_images" / "v1" / scene_name
    mv_paths = [str(mv_dir / f"{i:05d}.jpg") for i in range(start_idx, start_idx + 5 * 10, 5)]
    bev_folder = clips_dir / "BEV_images" / scene_name / f"clip_{start_idx}"
    bev_paths = sorted(str(f) for f in bev_folder.iterdir() if f.is_file())
    return mv_paths + bev_paths


def run_scene(llm, sampling_params, system_prompt, classification_prompt, clips_dir: Path, scene_name: str,
              run_dir: Path, model_name: str):
    output_dir = run_dir / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / CHECK_RESPONSE_NAME

    ego_status_path = clips_dir / "ego_status_in_text_enhanced" / f"{scene_name}.json"
    if not ego_status_path.exists():
        print(f"[SKIP] No ego_status for {scene_name}")
        return

    with open(ego_status_path, encoding="utf-8") as f:
        ego_status = json.load(f)

    bev_root = clips_dir / "BEV_images"
    total_clips = len(list((bev_root / scene_name).iterdir()))

    all_responses, done_idx = [], set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            all_responses = json.load(f)
        done_idx = {item["start_idx"] for item in all_responses}

    missing = [i for i in range(total_clips) if i not in done_idx]
    if not missing:
        print(f"[{scene_name[:40]:<40s}] already complete ({total_clips} clips)")
        return

    print(f"\n[{scene_name}]  {len(done_idx)}/{total_clips} done, {len(missing)} to go")
    by_idx = {item["start_idx"]: item for item in all_responses}

    for i, idx in enumerate(missing):
        if not (bev_root / scene_name / f"clip_{idx}").exists():
            print(f"  [{idx:3d}] SKIP — no BEV folder")
            continue
        try:
            t0 = time.time()
            messages = build_messages(system_prompt, classification_prompt,
                                      ego_status[idx]["content"][1]["text"],
                                      clip_image_paths(clips_dir, scene_name, idx))
            resp_data = ask_vlm(llm, sampling_params, messages, model_name)
            dt = time.time() - t0

            by_idx[idx] = {"scene_name": scene_name, "start_idx": idx, "response": resp_data}
            done_idx.add(idx)
            all_responses = [by_idx[k] for k in sorted(by_idx)]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_responses, f, ensure_ascii=False, indent=2)

            tok = resp_data.get("usage", {}).get("total_tokens", "?")
            label, _ = parse_label(resp_data["choices"][0]["message"]["content"])
            print(f"  [{idx:3d}] done  label={label:<14s}  tokens={tok}  {dt:.1f}s  ({len(missing)-i-1} left in scene)")
        except Exception as e:
            print(f"  [{idx:3d}] ERROR: {e}")

    print(f"  → {len(done_idx)}/{total_clips} clips saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips-dir", default=str(B2D_ROOT / "StyleClips"),
                        help="Output of the prepare_* scripts (multi_view_images/, BEV_images/, ego_status_in_text_enhanced/).")
    parser.add_argument("--results-dir", default=str(B2D_ROOT / "StyleResults"))
    parser.add_argument("--scenes-file", default=str(B2D_DEFAULT_SCENES_FILE),
                        help="JSON scene list ({'train': [...]} or flat list).")
    parser.add_argument("--scene", default=None, help="Only run scenes whose name contains this string.")
    parser.add_argument("--run-tag", default=None,
                        help="Run directory name to create or resume. Default: resume the latest *_v17_b2d run, else a new timestamped one.")
    add_vlm_args(parser)
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    results_dir = Path(args.results_dir)
    system_prompt = read_text(args.system_prompt_path)
    classification_prompt = read_text(args.prompt_path)

    if args.run_tag:
        run_dir = results_dir / args.run_tag
    else:
        candidates = sorted(results_dir.glob("*_v17_b2d"), reverse=True)
        if candidates:
            run_dir = candidates[0]
            print(f"Auto-resuming: {run_dir.name}")
        else:
            run_dir = results_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_v17_b2d"
            print(f"New run: {run_dir.name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    all_scenes = load_scenes_file(args.scenes_file)
    print(f"Loaded {len(all_scenes)} scenes from {args.scenes_file}")
    if args.scene:
        all_scenes = [s for s in all_scenes if args.scene.lower() in s.lower()]
        if not all_scenes:
            print(f"No scenes match '{args.scene}'")
            return

    llm, sampling_params = build_llm(args, [clips_dir])

    bev_root = clips_dir / "BEV_images"
    total_clips = sum(len(list((bev_root / s).iterdir())) for s in all_scenes)
    done_clips = 0
    for s in all_scenes:
        p = run_dir / s / CHECK_RESPONSE_NAME
        if p.exists():
            with open(p, encoding="utf-8") as f:
                done_clips += len(json.load(f))

    print(f"\nModel: {args.model}  |  Prompt: {args.prompt_path}  |  {len(all_scenes)} scenes  |  Run: {run_dir.name}")
    print(f"Overall: {done_clips}/{total_clips} clips done ({total_clips - done_clips} remaining)\n")

    for scene_name in all_scenes:
        run_scene(llm, sampling_params, system_prompt, classification_prompt, clips_dir, scene_name, run_dir,
                  args.model)

    print(f"\n{'='*60}")
    print(f"All scenes complete. Results in: {run_dir}")


if __name__ == "__main__":
    main()
