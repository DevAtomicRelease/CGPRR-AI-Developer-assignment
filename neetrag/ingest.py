"""Turn the shipped CSV tables and Markdown documents into citable chunks.

Each chunk id points at an exact place a reviewer can open:
  closing_ranks.csv#L57     -> line 57 of data/tables/closing_ranks.csv
  counselling_handbook.md#4-rounds -> the "4. Rounds" section of that file
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

QUOTA_LABEL = {"AIQ": "All India Quota (AIQ)", "State": "State Quota", "Deemed": "Deemed University quota"}


@dataclass
class Chunk:
    id: str
    source: str  # path relative to data/
    text: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _college_phrase(r: dict) -> str:
    return f"{r['college']} ({r['college_code']}), {r.get('city', '') + ', ' if r.get('city') else ''}{r['state']} ({r['college_type']} college)"


def _table_chunks(path: Path, rel: str) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line_no, r in enumerate(csv.DictReader(f), start=2):  # line 1 is the header
            meta = {k: r[k] for k in ("college_code", "state", "college_type", "course") if k in r}
            meta["year"] = int(r["year"])
            if "quota" in r:
                meta["quota"] = r["quota"]
            if "category" in r:
                meta["category"] = r["category"]
            if path.name == "closing_ranks.csv":
                meta.update(kind="closing_rank", round=int(r["round"]))
                text = (
                    f"Closing rank record (synthetic data). Year {r['year']}, Round {r['round']} counselling. "
                    f"College: {_college_phrase(r)}. Course: {r['course']}. Quota: {QUOTA_LABEL[r['quota']]}. "
                    f"Category: {r['category']}. Opening rank: {r['opening_rank']}. Closing rank: {r['closing_rank']}."
                )
            elif path.name == "seat_matrix.csv":
                meta.update(kind="seats")
                text = (
                    f"Seat matrix record (synthetic data). Year {r['year']}. College: {_college_phrase(r)}. "
                    f"Course: {r['course']}. Quota: {QUOTA_LABEL[r['quota']]}. Category: {r['category']}. "
                    f"Number of seats: {r['seats']}."
                )
            else:
                meta.update(kind="fees")
                text = (
                    f"Fee record for {r['year']} (synthetic data). College: {_college_phrase(r)}. Course: {r['course']}. "
                    f"Annual tuition fee: INR {r['annual_tuition_inr']}. Annual hostel fee: INR {r['annual_hostel_inr']}."
                )
            chunks.append(Chunk(id=f"{path.name}#L{line_no}", source=rel, text=text, meta=meta))
    return chunks


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def _markdown_chunks(path: Path, rel: str) -> list[Chunk]:
    """One chunk per '## ' section; the document title is prepended for context."""
    text = path.read_text(encoding="utf-8")
    title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), path.stem)
    parts = re.split(r"^## ", text, flags=re.M)
    chunks = []
    intro = parts[0].split("\n", 1)[1].strip() if "\n" in parts[0] else ""
    if intro:
        chunks.append(Chunk(id=f"{path.name}#intro", source=rel, text=f"{title}. {intro}", meta={"kind": "doc"}))
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        body = body.strip()
        if not body:
            continue
        chunks.append(
            Chunk(
                id=f"{path.name}#{_slug(heading)}",
                source=rel,
                text=f"{title} - {heading.strip()}\n{body}",
                meta={"kind": "doc", "heading": heading.strip()},
            )
        )
    return chunks


def load_chunks(data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted((data_dir / "tables").glob("*.csv")):
        chunks += _table_chunks(p, p.relative_to(data_dir).as_posix())
    for p in sorted((data_dir / "docs").glob("*.md")):
        chunks += _markdown_chunks(p, p.relative_to(data_dir).as_posix())
    return chunks
