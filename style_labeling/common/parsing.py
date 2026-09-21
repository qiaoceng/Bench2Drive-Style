"""Parse the free-text VLM answer into a label, confidence and reasoning fields.

Used by both datasets; the Bench2Drive training labels were produced with exactly these rules.
"""
import re


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<::>.*?</::>\s*", "", text, flags=re.DOTALL)
    return text


def parse_label(content_str: str):
    content_str = strip_thinking(content_str)
    m = re.findall(
        r"(?:^|\n)\s*(?:###?\s*)?Label\s*:\s*(aggressive|normal|conservative)\s*\|\s*(?:Confidence:\s*)?([0-9]*\.?[0-9]+)",
        content_str, re.IGNORECASE,
    )
    if m:
        return m[-1][0].lower(), m[-1][1]
    labels = re.findall(r"(?:^|\n)\s*(?:###?\s*)?Label\s*:\s*(aggressive|normal|conservative)", content_str, re.IGNORECASE)
    confs = re.findall(r"(?:^|\n)\s*(?:###?\s*)?Confidence\s*:\s*<?([0-9]*\.?[0-9]+)", content_str, re.IGNORECASE)
    if labels:
        return labels[-1].lower(), confs[-1] if confs else ""
    init = re.findall(r"Initial Label\s*:\s*(aggressive|normal|conservative)", content_str, re.IGNORECASE)
    init_conf = re.findall(r"Initial Confidence\s*:\s*<?([0-9]*\.?[0-9]+)", content_str, re.IGNORECASE)
    if init:
        return init[-1].lower(), init_conf[-1] if init_conf else ""
    return "unknown", ""


def parse_description(content_str: str) -> str:
    content_str = strip_thinking(content_str)
    matches = re.findall(r"(?:^|\n)\s*Description\s*:\s*(.+)", content_str, re.IGNORECASE)
    return matches[-1].strip() if matches else ""


def parse_datasummary(content_str: str) -> str:
    ds = re.search(r"(?:STAGE\s*0|SCENE\s*DECOMPOSITION).*?\n(.*?)(?=STAGE\s*1|INITIAL)", content_str, re.IGNORECASE | re.DOTALL)
    return ds.group(1).strip() if ds else ""


def parse_reasoning(content_str: str) -> dict:
    reasoning = {}
    stage3 = re.search(r"STAGE\s*3.*?FINAL.*?\n(.*)", content_str, re.DOTALL | re.IGNORECASE)
    search_text = stage3.group(1) if stage3 else content_str
    for i in range(1, 8):
        next_markers = [rf"\nA{i + 1}\."] if i < 7 else []
        next_markers += [r"\n\s*Initial Label", r"\n\s*###", r"\n\s*STAGE", r"\n\s*Label"]
        stop = "|".join(next_markers) if next_markers else r"$"
        pattern = rf"A{i}\.\s*(.*?)(?={stop}|$)"
        m = re.search(pattern, search_text, re.DOTALL | re.IGNORECASE)
        if m:
            reasoning[f"A{i}"] = m.group(1).strip()
    return reasoning


def parse_scene_decomposition(content_str: str) -> dict:
    decomp = {}
    for tag in ["V1", "V2", "V3", "V4", "V5", "T1", "T2", "T3", "T4", "B1", "B2"]:
        m = re.search(rf"{tag}[.:]\s*(.+?)(?=\n\s*[VTB]\d|$|\n\s*\[|\n\s*###|\n\s*STAGE)", content_str, re.IGNORECASE)
        if m:
            decomp[tag] = m.group(1).strip()
    return decomp


def _find(s: str, needle: str, start: int = 0) -> int:
    return s.lower().find(needle.lower(), start)


def parse_alternatives(content_str: str) -> dict:
    """Slice-based to avoid catastrophic backtracking on runaway model output."""
    alt = {}
    a_pos = _find(content_str, "Path A")
    b_pos = _find(content_str, "Path B", a_pos + 1) if a_pos >= 0 else -1
    c_pos = _find(content_str, "Path C", b_pos + 1) if b_pos >= 0 else -1
    r_pos = _find(content_str, "Resolution", max(a_pos, b_pos, c_pos) + 1)

    if a_pos >= 0 and b_pos > a_pos:
        seg = content_str[a_pos:b_pos]
        m = re.search(r"[\"“”]([^\"“”\n]{1,500})[\"“”]\??\s*(.{0,500})", seg, re.DOTALL)
        if m:
            alt["PathA_label"] = m.group(1).strip()
            alt["PathA_argument"] = m.group(2).strip()[:200]
    if b_pos >= 0:
        end = c_pos if c_pos > b_pos else (r_pos if r_pos > b_pos else min(b_pos + 2000, len(content_str)))
        seg = content_str[b_pos:end]
        m = re.search(r"\n(.{0,1000})", seg, re.DOTALL)
        if m:
            alt["PathB_argument"] = m.group(1).strip()[:200]
    if c_pos >= 0:
        end = r_pos if r_pos > c_pos else min(c_pos + 2000, len(content_str))
        seg = content_str[c_pos:end]
        m = re.search(r"\n(.{0,1500})", seg, re.DOTALL)
        if m:
            alt["PathC_temporal"] = m.group(1).strip()[:300]
    if r_pos >= 0:
        seg = content_str[r_pos:r_pos + 2000]
        m = re.search(r"Resolution[.:]\s*(.{0,1000})", seg, re.DOTALL | re.IGNORECASE)
        if m:
            alt["Resolution"] = m.group(1).strip()[:200]
    return alt


def parse_response(content_str: str) -> dict:
    """Parse one VLM answer into the record stored in parsed_reasons_vllm.json (minus scene/index/usage)."""
    label, conf_str = parse_label(content_str)
    try:
        conf = float(conf_str) if conf_str else None
    except ValueError:
        conf = None
    desc = parse_description(content_str)
    return {
        "label": label,
        "confidence": conf,
        "description": desc,
        "content": {
            "SceneDecomposition": parse_scene_decomposition(content_str),
            "DataSummary": parse_datasummary(content_str),
            "Reasoning": parse_reasoning(content_str),
            "Alternatives": parse_alternatives(content_str),
            "Label": label,
            "Confidence": conf,
            "Description": desc,
        },
    }
