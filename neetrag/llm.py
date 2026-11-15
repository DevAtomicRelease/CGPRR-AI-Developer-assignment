"""Thin clients for the local model server (Ollama by default, or any local
OpenAI-compatible server such as vLLM). Nothing here calls a cloud service."""

from __future__ import annotations

import json
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
        raise LLMError(f"Cannot reach local model server at {url}: {e}") from e
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

def chat_json(system: str, user: str, schema: dict[str, Any]) -> dict:
    """Ask the model for a JSON object that follows `schema`."""
    if SETTINGS.llm_backend == "openai":
        payload = {
            "model": SETTINGS.chat_model,
            "temperature": SETTINGS.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
        }
        data = _post(f"{SETTINGS.openai_url}/chat/completions", payload)
        text = data["choices"][0]["message"]["content"]
    else:
        payload = {
            "model": SETTINGS.chat_model,
            "stream": False,
            "format": schema,
            "options": {"temperature": SETTINGS.temperature, "num_ctx": SETTINGS.num_ctx, "seed": 7},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        data = _post(f"{SETTINGS.ollama_url}/api/chat", payload)
        text = data["message"]["content"]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model did not return valid JSON: {text[:300]}") from e


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
