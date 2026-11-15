"""Rule-based extraction of counselling entities from a question.

Used for metadata filtering during retrieval, for routing rank questions to the
predictor, and for deterministic "not in the sources" checks (unknown college,
year outside the data). Rules are deliberately simple and auditable.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import SETTINGS

GENERIC_WORDS = [
    "government medical college", "deemed university medical college", "institute of medical sciences",
    "private medical college", "medical college",
]

CATEGORY_PATTERNS = {
    "GEN": r"\b(gen|general|unreserved|ur|open category)\b",
    "EWS": r"\bews\b|economically weaker",
    "OBC": r"\bobc(?:-ncl)?\b|other backward",
    "SC": r"\bsc\b|scheduled caste",
    "ST": r"\bst\b|scheduled tribe",
}
QUOTA_PATTERNS = {
    "AIQ": r"\baiq\b|all[- ]india quota|all india",
    "State": r"state quota|domicile",
    "Deemed": r"\bdeemed\b",
}
# Things people ask about that this dataset does not cover at all.
UNCOVERED_PATTERNS = {
    "NRI quota": r"\bnri\b",
    "management quota": r"management quota",
    "PwD quota": r"\bpwd\b|\bph\b quota|disab",
    "mop-up / stray round": r"mop[- ]?up|stray",
    "AYUSH / nursing courses": r"\b(bams|bhms|bums|nursing|bsc|bpt|bvsc)\b",
    "PG / NEET-PG counselling": r"\bneet[- ]?pg\b|\bmd\b|\bms\b seat|post ?graduate",
}
INSTITUTION_RE = re.compile(
    r"\b((?i:aiims|jipmer|afmc|cmc vellore|kgmu)(?:\s+[A-Z][\w.'&-]*)?"
    r"|[A-Z][\w.'&-]*(?:\s+[A-Z][\w.'&-]*)*\s+"
    r"(?i:medical college|institute of medical sciences|medical university|dental college|hospital))"
)


@dataclass
class Vocab:
    colleges: dict[str, str]  # code -> full name
    aliases: dict[str, str]  # lowercase alias -> code
    states: set[str]
    years: list[int]
    rounds: list[int]
    courses: set[str]


@lru_cache(maxsize=1)
def vocab(data_dir: Path | None = None) -> Vocab:
    path = (data_dir or SETTINGS.data_dir) / "tables" / "closing_ranks.csv"
    colleges, states, years, rounds, courses = {}, set(), set(), set(), set()
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            colleges[r["college_code"]] = r["college"]
            states.add(r["state"])
            years.add(int(r["year"]))
            rounds.add(int(r["round"]))
            courses.add(r["course"])
    aliases = {}
    for code, name in colleges.items():
        low = name.lower()
        aliases[low] = code
        aliases[code.lower()] = code
        short = low
        for g in GENERIC_WORDS:
            short = short.replace(g, "")
        short = short.strip()
        if len(short) >= 5:
            aliases[short] = code
    return Vocab(colleges, aliases, states, sorted(years), sorted(rounds), courses)


@dataclass
class Entities:
    colleges: list[str] = field(default_factory=list)
    unknown_institutions: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    rounds: list[int] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    quotas: list[str] = field(default_factory=list)
    courses: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    rank: int | None = None
    uncovered: list[str] = field(default_factory=list)

    def filters(self) -> dict[str, list]:
        """Metadata filters for table rows (empty list = no constraint)."""
        f: dict[str, list] = {}
        if self.colleges:
            f["college_code"] = self.colleges
        if self.years:
            f["year"] = self.years
        if self.rounds:
            f["round"] = self.rounds
        if self.categories:
            f["category"] = self.categories
        if self.quotas:
            f["quota"] = self.quotas
        if self.courses:
            f["course"] = self.courses
        return f


# The user's OWN rank ("my rank is 12000", "with AIR 8000", "12000 rank"), as opposed
# to a closing rank quoted in a question ("the closing rank is 50, right?").
OWN_RANK_RE = re.compile(
    r"\b(?:my|i have|i got|i scored|i am at|with|having)\b[^.?!\d]{0,25}\b(?:rank|air)\b[^.?!\d]{0,12}(\d[\d,]*)"
    r"|\b(?:my )?air\s*(?:is|of|=|:)?\s*(\d[\d,]*)"
    r"|(\d[\d,]*)\s*(?:rank|air)\b",
    re.I,
)
GENERIC_RANK_RE = re.compile(r"\b(?:rank|air)\s*(?:is|of|=|:|around|about)?\s*(\d[\d,]*)", re.I)


def parse_rank(q: str, generic: bool = False) -> int | None:
    m = OWN_RANK_RE.search(q) or (GENERIC_RANK_RE.search(q) if generic else None)
    if not m:
        return None
    raw = next(g for g in m.groups() if g).replace(",", "")
    return int(raw) if raw.isdigit() else None


def extract(q: str) -> Entities:
    v = vocab()
    low = q.lower()
    e = Entities()

    for alias, code in sorted(v.aliases.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b" + re.escape(alias) + r"\b", low) and code not in e.colleges:
            e.colleges.append(code)
    # Institution-like phrases that do not match any known college.
    for m in INSTITUTION_RE.finditer(q):
        phrase = m.group(0).strip()
        pl = phrase.lower()
        if not any(alias in pl or pl in alias for alias in v.aliases if len(alias) >= 5):
            e.unknown_institutions.append(phrase)

    e.years = sorted({int(y) for y in re.findall(r"\b(20\d\d|19\d\d)\b", q)})
    e.rounds = sorted({int(x) for x in re.findall(r"round\s*(\d)", low)})
    if re.search(r"\bfirst round\b", low):
        e.rounds.append(1)
    if re.search(r"\bsecond round\b", low):
        e.rounds.append(2)
    e.categories = [c for c, p in CATEGORY_PATTERNS.items() if re.search(p, low)]
    e.quotas = [qn for qn, p in QUOTA_PATTERNS.items() if re.search(p, low)]
    e.courses = [c for c in sorted(v.courses) if re.search(r"\b" + c.lower() + r"\b", low)]
    e.states = [s for s in sorted(v.states) if s.lower() in low]
    e.rank = parse_rank(q)
    e.uncovered = [name for name, p in UNCOVERED_PATTERNS.items() if re.search(p, low)]
    return e
