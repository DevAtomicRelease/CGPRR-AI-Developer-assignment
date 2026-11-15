"""Runtime settings. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(_env("NEETRAG_DATA_DIR", str(ROOT / "data")))
    index_dir: Path = Path(_env("NEETRAG_INDEX_DIR", str(ROOT / "index")))
    log_dir: Path = Path(_env("NEETRAG_LOG_DIR", str(ROOT / "logs")))

    # LLM backend: "ollama" (default) or "openai" (any local OpenAI-compatible
    # server such as vLLM or llama.cpp). Both run on the local machine.
    llm_backend: str = _env("NEETRAG_LLM_BACKEND", "ollama")
    # Embeddings stay on Ollama by default even when chat runs on vLLM, because a vLLM
    # server started for a chat model does not serve an embedding model.
    embed_backend: str = _env("NEETRAG_EMBED_BACKEND", "ollama")
    ollama_url: str = _env("OLLAMA_URL", "http://127.0.0.1:11434")
    openai_url: str = _env("NEETRAG_OPENAI_URL", "http://127.0.0.1:8000/v1")
    chat_model: str = _env("NEETRAG_CHAT_MODEL", "qwen2.5:7b-instruct")
    embed_model: str = _env("NEETRAG_EMBED_MODEL", "nomic-embed-text")

    top_k: int = int(_env("NEETRAG_TOP_K", "8"))
    # Minimum cosine similarity of the best dense hit; below it we refuse
    # without calling the LLM.
    min_dense_score: float = float(_env("NEETRAG_MIN_DENSE_SCORE", "0.60"))
    temperature: float = float(_env("NEETRAG_TEMPERATURE", "0.0"))
    num_ctx: int = int(_env("NEETRAG_NUM_CTX", "8192"))
    request_timeout: int = int(_env("NEETRAG_TIMEOUT", "180"))


SETTINGS = Settings()
