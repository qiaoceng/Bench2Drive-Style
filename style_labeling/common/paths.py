"""Default locations inside the repo. Every script takes CLI flags that override these."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

B2D_ROOT = DATA_ROOT / "Bench2Drive"
SD_ROOT = DATA_ROOT / "StyleDrive"

PROMPT_DIR = REPO_ROOT / "style_labeling" / "prompts"
SYSTEM_PROMPT_PATH = PROMPT_DIR / "system_prompt.txt"
CLASSIFICATION_PROMPT_PATH = PROMPT_DIR / "classification_prompt.txt"

B2D_DEFAULT_SCENES_FILE = REPO_ROOT / "style_labeling" / "bench2drive" / "splits" / "labeled_scenes.json"


def load_scenes_file(path) -> List[str]:
    """Load a scene list from {"train": ["v1/SceneName", ...]} or a flat list; the 'v1/' prefix is stripped."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    raw = obj.get("train", obj) if isinstance(obj, dict) else obj
    out = []
    for s in raw:
        # strip the "v1/" prefix (also tolerates "v1//Foo")
        s = re.sub(r"^v\d+/+", "", s).strip()
        if s:
            out.append(s)
    return out
