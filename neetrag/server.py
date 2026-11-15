"""Local HTTP API and single-page web UI.

  GET  /               web UI
  POST /api/ask        {"question": "..."}
  POST /api/predict    {"rank": 12000, "category": "OBC", "state": "West Bengal", "course": "MBBS"}
  GET  /api/health     model + index status
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .assistant import ask, get_index
from .llm import server_info
from .predictor import predict

app = FastAPI(title="NEET.info private RAG assistant", version="0.1.0")
STATIC = Path(__file__).parent / "static"


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class PredictIn(BaseModel):
    rank: int
    category: str
    course: str = "MBBS"
    state: str | None = None
    year: int | None = None
    round: int | None = None


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    idx = get_index()
    return {"index": idx.manifest, "model": server_info()}


@app.post("/api/ask")
def api_ask(body: AskIn) -> dict:
    return ask(body.question)


@app.post("/api/predict")
def api_predict(body: PredictIn) -> dict:
    return asdict(predict(body.rank, body.category, body.course, body.state, body.year, body.round))
