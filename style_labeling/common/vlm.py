"""vLLM engine setup and a single classification request, shared by both datasets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .parsing import strip_thinking
from .paths import CLASSIFICATION_PROMPT_PATH, SYSTEM_PROMPT_PATH

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
CHECK_RESPONSE_NAME = "check_response.json"


def add_vlm_args(parser, default_max_model_len: int = 65536, default_tp: int = 2):
    """CLI flags for the prompt files and engine. Defaults reproduce the v17 runs used in the paper results."""
    parser.add_argument("--prompt-path", default=str(CLASSIFICATION_PROMPT_PATH),
                        help="Classification prompt (.txt).")
    parser.add_argument("--system-prompt-path", default=str(SYSTEM_PROMPT_PATH),
                        help="System prompt (.txt).")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--tp", type=int, default=default_tp, help="tensor_parallel_size")
    parser.add_argument("--max-model-len", type=int, default=default_max_model_len)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)


def read_text(path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_llm(args, media_dirs: List[Path]):
    """Create the vLLM engine and sampling params. `media_dirs` must contain every image the prompts reference."""
    from vllm import LLM, SamplingParams

    allowed = os.path.commonpath([str(Path(d).resolve()) for d in media_dirs])
    print(f"Loading vLLM engine: {args.model}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
          f"tp={args.tp}, max_model_len={args.max_model_len})", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tp,
        limit_mm_per_prompt={"image": 20},
        trust_remote_code=True,
        allowed_local_media_path=allowed,
        mm_processor_kwargs={
            "min_pixels": 28 * 28 * 4,
            "max_pixels": 28 * 28 * 480,
        },
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_model_len,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )
    print("vLLM engine loaded.", flush=True)
    return llm, sampling_params


def build_messages(system_prompt: str, classification_prompt: str, ego_text: str, image_paths: List[str]) -> list:
    """Classification prompt, then the ego-status text, then the images (multi-view frames followed by BEV frames)."""
    user_content = [
        {"type": "text", "text": classification_prompt},
        {"type": "text", "text": ego_text},
    ]
    for p in image_paths:
        user_content.append({"type": "image_url", "image_url": {"url": f"file://{Path(p).resolve()}"}})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def ask_vlm(llm, sampling_params, messages: list, model_name: str = MODEL_NAME) -> dict:
    """Run a single inference; return an OpenAI-shaped response dict."""
    out = llm.chat(messages=messages, sampling_params=sampling_params, use_tqdm=False)[0]
    content_str = strip_thinking(out.outputs[0].text)
    return {
        "id": out.request_id,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content_str},
            "finish_reason": out.outputs[0].finish_reason,
        }],
        "usage": {
            "prompt_tokens": len(out.prompt_token_ids),
            "completion_tokens": len(out.outputs[0].token_ids),
            "total_tokens": len(out.prompt_token_ids) + len(out.outputs[0].token_ids),
        },
    }
