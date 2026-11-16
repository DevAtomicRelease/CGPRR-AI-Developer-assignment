"""Thin clients for the local model server (Ollama by default, or any local
OpenAI-compatible server such as vLLM). Nothing here calls a cloud service."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import requests

from .config import SETTINGS


class LLMError(RuntimeError):
    pass


def _post(url: str, payload: dict) -> dict:
    try:
        r = requests.post(url, json=payload, timeout=SETTINGS.request_timeout)
    except requests.RequestException as e:  # pragma: no cover - network failure path
        raise LLMError(f"Cannot reach local model server at {url} ({type(e).__name__})") from e
    if r.status_code != 200:
        raise LLMError(f"{url} returned {r.status_code}: {r.text[:300]}")
    return r.json()


# ---------------------------------------------------------------- embeddings

def embed(texts: list[str], kind: str = "document", batch: int = 64) -> np.ndarray:
    """Return L2-normalised embeddings. nomic-embed-text expects task prefixes."""
    prefix = "search_query: " if kind == "query" else "search_document: "
    if "nomic" not in SETTINGS.embed_model:
        prefix = ""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [prefix + t for t in texts[i : i + batch]]
        if SETTINGS.embed_backend == "openai":
            data = _post(f"{SETTINGS.openai_url}/embeddings", {"model": SETTINGS.embed_model, "input": chunk})
            out.extend(d["embedding"] for d in data["data"])
        else:
            data = _post(f"{SETTINGS.ollama_url}/api/embed", {"model": SETTINGS.embed_model, "input": chunk})
            out.extend(data["embeddings"])
    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-9, None)


# ---------------------------------------------------------------------- chat

class LLMMalformed(LLMError):
    """The model hit the token cap or returned invalid JSON (for example a repetition loop).
    `partial` holds the text generated before generation stopped."""

    def __init__(self, msg: str, partial: str = ""):
        super().__init__(msg)
        self.partial = partial


# Five or more citation markers in a row ("[S1] [S2] [S3] [S4] [S5]") is the degenerate
# loop seen in testing; stop generating as soon as it appears instead of waiting for the cap.
CITATION_LOOP = re.compile(r"(?:\[S\d+\]\s*){5,}$")


def _stream_ollama(payload: dict) -> tuple[str, bool]:
    url = f"{SETTINGS.ollama_url}/api/chat"
    parts: list[str] = []
    try:
        with requests.post(url, json=payload, stream=True, timeout=SETTINGS.request_timeout) as r:
            if r.status_code != 200:
                raise LLMError(f"{url} returned {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise LLMError(f"{url}: {msg['error']}")
                parts.append(msg.get("message", {}).get("content", ""))
                if CITATION_LOOP.search("".join(parts[-40:])):
                    raise LLMMalformed("Model started repeating citation markers; generation stopped early.",
                                       partial="".join(parts))
                if msg.get("done"):
                    return "".join(parts), msg.get("done_reason") == "length"
    except requests.RequestException as e:
        raise LLMError(f"Cannot reach local model server at {url} ({type(e).__name__})") from e
    return "".join(parts), True


def chat_json(system: str, user: str, schema: dict[str, Any], retry: bool = False) -> dict:
    """Ask the model for a JSON object that follows `schema`.

    Output is capped at SETTINGS.max_tokens so a degenerate repetition loop costs seconds,
    not minutes. `retry=True` is the second attempt: a little sampling and a repetition
    penalty to get the model out of the loop it fell into.
    """
    temperature = 0.3 if retry else SETTINGS.temperature
    if SETTINGS.llm_backend == "openai":
        payload = {
            "model": SETTINGS.chat_model,
            "temperature": temperature,
            "max_tokens": SETTINGS.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
        }
        if retry:
            payload["frequency_penalty"] = 0.5
        data = _post(f"{SETTINGS.openai_url}/chat/completions", payload)
        text = data["choices"][0]["message"]["content"]
        truncated = data["choices"][0].get("finish_reason") == "length"
    else:
        options = {"temperature": temperature, "num_ctx": SETTINGS.num_ctx, "num_predict": SETTINGS.max_tokens,
                   "seed": 11 if retry else 7}
        if retry:
            options["repeat_penalty"] = 1.3
        payload = {
            "model": SETTINGS.chat_model,
            "stream": True,
            "format": schema,
            "options": options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        text, truncated = _stream_ollama(payload)
    if truncated:
        raise LLMMalformed(f"Model output hit the {SETTINGS.max_tokens}-token cap: ...{text[-120:]}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMMalformed(f"Model did not return valid JSON: {text[:300]}") from e


def server_info() -> dict:
    """Best-effort description of the running model, for logs and the UI."""
    info = {"backend": SETTINGS.llm_backend, "chat_model": SETTINGS.chat_model, "embed_model": SETTINGS.embed_model}
    if SETTINGS.llm_backend == "ollama":
        try:
            ps = requests.get(f"{SETTINGS.ollama_url}/api/ps", timeout=5).json()
            info["loaded"] = [
                {"name": m["name"], "size_vram_gb": round(m.get("size_vram", 0) / 1e9, 2)} for m in ps.get("models", [])
            ]
        except requests.RequestException:
            info["loaded"] = "unreachable"
    return info
