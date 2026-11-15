"""Measure local generation speed and VRAM for the design note.

  python scripts/bench.py            # 5 runs of a typical RAG prompt
Prints prompt-processing and generation tokens/second (from Ollama's own timers)
and the VRAM held by each loaded model.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neetrag import assistant as a  # noqa: E402
from neetrag.config import SETTINGS  # noqa: E402
from neetrag.entities import extract  # noqa: E402

QUESTIONS = [
    "What was the 2025 Round 2 AIQ GEN closing rank for MBBS at Ganga Valley Government Medical College?",
    "What documents do I need to carry when reporting to the allotted college?",
    "What is the annual tuition fee for MBBS at Salt Lake Institute of Medical Sciences?",
    "How many State Quota OBC MBBS seats did Kalinga Coast Government Medical College have in 2025?",
    "What is the security deposit if I include deemed university seats in my choices?",
]


def main() -> None:
    rows = []
    for q in QUESTIONS:
        hits = a.get_index().search(q, filters=extract(q).filters())
        src = "\n".join(f"[S{i}] ({h.chunk.id}) {h.chunk.text}" for i, h in enumerate(hits, 1))
        # keep_alive + a unique system suffix stops Ollama reusing the cached prompt, so the
        # prompt-processing figure is a cold one.
        r = requests.post(f"{SETTINGS.ollama_url}/api/chat", json={
            "model": SETTINGS.chat_model, "stream": False, "format": a.ANSWER_SCHEMA,
            "options": {"temperature": 0, "num_ctx": SETTINGS.num_ctx},
            "messages": [{"role": "system", "content": a.SYSTEM_PROMPT + f"\n(run {len(rows)})"},
                         {"role": "user", "content": f"SOURCES:\n{src}\n\nQUESTION: {q}"}],
        }, timeout=300).json()
        rows.append({
            "prompt_tokens": r["prompt_eval_count"],
            "prompt_tps": r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9),
            "gen_tokens": r["eval_count"],
            "gen_tps": r["eval_count"] / (r["eval_duration"] / 1e9),
            "total_s": r["total_duration"] / 1e9,
        })
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
    print(f"model {SETTINGS.chat_model}, {len(rows)} runs (median):")
    print(f"  prompt tokens {med['prompt_tokens']:.0f} @ {med['prompt_tps']:.0f} tok/s")
    print(f"  output tokens {med['gen_tokens']:.0f} @ {med['gen_tps']:.1f} tok/s")
    print(f"  end-to-end generation {med['total_s']:.2f} s")
    for m in requests.get(f"{SETTINGS.ollama_url}/api/ps", timeout=5).json().get("models", []):
        print(f"  VRAM {m['name']}: {m.get('size_vram', 0) / 2**30:.2f} GiB "
              f"({m['details'].get('quantization_level')}, ctx {m.get('context_length')})")


if __name__ == "__main__":
    main()
