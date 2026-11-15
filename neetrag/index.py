"""Hybrid retrieval index: dense embeddings (local model) + BM25, fused with
reciprocal-rank fusion, with optional metadata filters for table rows.

Stored on local disk under index/:
  chunks.jsonl    the text and metadata of every chunk
  embeddings.npy  float32 matrix, one L2-normalised row per chunk
  manifest.json   models used, chunk count, data fingerprint
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SETTINGS
from .ingest import Chunk, load_chunks
from .llm import embed

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str], ids: np.ndarray) -> np.ndarray:
        out = np.zeros(len(ids), dtype=np.float32)
        q = [t for t in set(query) if t in self.idf]
        for j, i in enumerate(ids):
            tf, dl = self.tf[i], self.len[i]
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[j] = s
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused RRF score
    dense: float  # cosine similarity
    lexical: float  # BM25 score


def _fingerprint(data_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(data_dir.rglob("*")):
        if p.is_file() and p.suffix in {".csv", ".md"}:
            h.update(p.relative_to(data_dir).as_posix().encode())
            # Normalise line endings so a Windows and a Linux checkout give the same fingerprint.
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def build_index(data_dir: Path | None = None, index_dir: Path | None = None) -> dict:
    data_dir = data_dir or SETTINGS.data_dir
    index_dir = index_dir or SETTINGS.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(data_dir)
    t0 = time.time()
    vecs = embed([c.text for c in chunks], kind="document")
    secs = time.time() - t0
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    np.save(index_dir / "embeddings.npy", vecs)
    manifest = {
        "chunks": len(chunks),
        "dim": int(vecs.shape[1]),
        "embed_model": SETTINGS.embed_model,
        "data_fingerprint": _fingerprint(data_dir),
        "embed_seconds": round(secs, 1),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class Index:
    def __init__(self, index_dir: Path | None = None):
        index_dir = index_dir or SETTINGS.index_dir
        if not (index_dir / "embeddings.npy").exists():
            raise FileNotFoundError(f"No index in {index_dir}. Run: python -m neetrag build-index")
        self.chunks = [Chunk(**json.loads(ln)) for ln in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.vecs = np.load(index_dir / "embeddings.npy")
        self.manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("embed_model") != SETTINGS.embed_model:
            raise RuntimeError(f"Index was built with embedding model {self.manifest.get('embed_model')!r} but "
                               f"{SETTINGS.embed_model!r} is configured. Run: python -m neetrag build-index")
        if self.manifest.get("data_fingerprint") != _fingerprint(SETTINGS.data_dir):
            warnings.warn("data/ has changed since the index was built; answers may cite stale chunks. "
                          "Run: python -m neetrag build-index", stacklevel=2)
        self.bm25 = BM25([tokenize(c.text) for c in self.chunks])
        self.by_id = {c.id: c for c in self.chunks}

    def _candidates(self, filters: dict[str, list]) -> np.ndarray:
        """Documents always pass; table rows must match every filter they carry."""
        keep = []
        for i, c in enumerate(self.chunks):
            if c.meta.get("kind") != "doc":
                ok = all(c.meta.get(k) in vals for k, vals in filters.items() if k in c.meta)
                if not ok:
                    continue
            keep.append(i)
        return np.array(keep, dtype=np.int64)

    def search(self, query: str, k: int | None = None, filters: dict[str, list] | None = None) -> list[Hit]:
        k = k or SETTINGS.top_k
        ids = self._candidates(filters or {})
        if len(ids) == 0:
            return []
        qv = embed([query], kind="query")[0]
        dense = self.vecs[ids] @ qv
        lex = self.bm25.scores(tokenize(query), ids)
        # Reciprocal-rank fusion (k=60) of the two rankings.
        rrf = np.zeros(len(ids), dtype=np.float32)
        for scores in (dense, lex):
            order = np.argsort(-scores)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(order))
            rrf += 1.0 / (60 + ranks + 1)
        top = np.argsort(-rrf)[:k]
        return [Hit(self.chunks[ids[j]], float(rrf[j]), float(dense[j]), float(lex[j])) for j in top]
