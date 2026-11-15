"""Deterministic safety checks that run before and after the language model."""

from __future__ import annotations

import json
import re
import time

from .config import SETTINGS

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b[^.?!]{0,40}\b(document|source|instruction|context|rule|data|table|prompt)s?\b",
    r"\byou are now\b|\bact as\b|\bpretend\b|\bjailbreak\b|\bsystem prompt\b|\bdeveloper mode\b",
    r"\b(tell|say|answer|confirm|state)\b[^.?!]{0,30}\b(that )?the\b[^.?!]{0,40}\b(cut-?off|closing rank|rank|fee|seats?)\b[^.?!]{0,20}\bis\b\s*\d",
    r"\bdo not (use|read|cite|check)\b[^.?!]{0,30}\b(source|document|data)s?\b",
]
GUARANTEE_PATTERNS = [
    r"\bguarantee", r"\bwill i (definitely |surely )?get\b", r"\bsure (to|of) get", r"\b100 ?%",
    r"\bshould i (choose|take|join|go|pick|opt)", r"\bchances?\b", r"\bcan i get\b", r"\bbest (college|option|choice)\b",
]
OPTIONS_INTENT = re.compile(
    r"which (college|colleges|options)|what (college|colleges|options)|can i get|options|colleges? (can|could|might)|"
    r"predict|should i (choose|take|join|go|pick|opt)|shortlist|where can i|which\b[^.?!]{0,25}\bcolleges?\b|possible|eligible for which",
    re.I,
)
NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def find_injection(text: str) -> list[str]:
    return [m.group(0) for p in INJECTION_PATTERNS for m in re.finditer(p, text, re.I)]


def strip_injection(text: str) -> str:
    """Drop every sentence that contains an injection pattern."""
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    kept = [s for s in sentences if not find_injection(s)]
    return " ".join(kept).strip()


def wants_guarantee(text: str) -> bool:
    return any(re.search(p, text, re.I) for p in GUARANTEE_PATTERNS)


def numbers(text: str, min_value: float = 10) -> set[str]:
    """Numbers that matter for grounding (ranks, seats, fees, years); small
    ordinals such as 'Round 1' are ignored."""
    out = set()
    for m in NUM_RE.findall(text):
        n = m.replace(",", "").rstrip(".")
        try:
            if float(n) >= min_value:
                out.add(n.rstrip("0").rstrip(".") if "." in n else n)
        except ValueError:
            continue
    return out


def log_event(kind: str, record: dict) -> None:
    """Append-only JSON-lines log on local disk (logs/). Nothing leaves the machine."""
    SETTINGS.log_dir.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **record}
    name = "security.jsonl" if kind in {"prompt_injection", "grounding_block", "policy_block"} else "queries.jsonl"
    with (SETTINGS.log_dir / name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
