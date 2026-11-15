"""Track A: rank-based college option finder for NEET.info.

Deterministic by design: every rank, band and fee shown comes straight from a
cited CSV row. No language model is involved in choosing options or numbers, so
the predictor cannot "invent" a cut-off. If the data needed for a request does
not exist (a year, a category, a course), it refuses instead of estimating.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache

from .config import SETTINGS

CATEGORIES = ["GEN", "EWS", "OBC", "SC", "ST"]
BANDS = [("Likely", 0.85), ("Borderline", 1.0), ("Reach", 1.15)]
MAX_RANK = 2_500_000
CAVEAT = (
    "These options come from past (synthetic) closing ranks. Cut-offs change every year with seats, "
    "candidates and choices, so this is not a prediction of allotment and not a guarantee of admission. "
    "Please confirm with the official counselling authority and a counsellor before filling choices."
)


@dataclass
class Row:
    ref: str
    data: dict


@lru_cache(maxsize=1)
def _tables() -> tuple[list[Row], dict[tuple, Row]]:
    rows = []
    with (SETTINGS.data_dir / "tables" / "closing_ranks.csv").open(encoding="utf-8") as f:
        for line, r in enumerate(csv.DictReader(f), start=2):
            for k in ("year", "round", "opening_rank", "closing_rank"):
                r[k] = int(r[k])
            rows.append(Row(f"closing_ranks.csv#L{line}", r))
    fees = {}
    with (SETTINGS.data_dir / "tables" / "fees_2025.csv").open(encoding="utf-8") as f:
        for line, r in enumerate(csv.DictReader(f), start=2):
            fees[(r["college_code"], r["course"])] = Row(f"fees_2025.csv#L{line}", r)
    return rows, fees


@dataclass
class Option:
    band: str
    college: str
    college_code: str
    state: str
    college_type: str
    course: str
    quota: str
    category_used: str
    closing_rank: int
    reason: str
    trend: list[dict]
    annual_tuition_inr: int | None
    citations: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    status: str  # "ok" | "refused"
    message: str
    inputs: dict
    options: list[Option] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    caveat: str = CAVEAT


def _refuse(msg: str, inputs: dict) -> Prediction:
    return Prediction(status="refused", message=msg, inputs=inputs, caveat="")


def predict(
    rank: int,
    category: str,
    course: str = "MBBS",
    domicile_state: str | None = None,
    year: int | None = None,
    round_: int | None = None,
    limit: int = 15,
    colleges: list[str] | None = None,
    quotas: list[str] | None = None,
) -> Prediction:
    """List options for a rank. If `colleges` (codes) is given, report on those
    colleges only, including ones the rank is outside of."""
    rows, fees = _tables()
    years = sorted({r.data["year"] for r in rows})
    rounds = sorted({r.data["round"] for r in rows})
    states = sorted({r.data["state"] for r in rows})
    courses = sorted({r.data["course"] for r in rows})
    category = (category or "").upper().strip()
    course = (course or "MBBS").upper().strip()
    inputs = {"rank": rank, "category": category, "course": course, "domicile_state": domicile_state,
              "year": year, "round": round_}

    # ---- refuse instead of guessing
    if not isinstance(rank, int) or rank < 1 or rank > MAX_RANK:
        return _refuse(f"Rank must be a NEET All India Rank between 1 and {MAX_RANK:,}.", inputs)
    if category not in CATEGORIES:
        return _refuse(f"There is no cut-off data for category '{category}'. The sources cover {', '.join(CATEGORIES)} only, "
                       "and I will not estimate cut-offs for other categories or sub-quotas.", inputs)
    if course not in courses:
        return _refuse(f"There is no cut-off data for {course}. The sources cover {', '.join(courses)} only.", inputs)
    if year is not None and year not in years:
        return _refuse(f"There are no closing ranks for {year} in the sources (they cover {years[0]}-{years[-1]}). "
                       "I will not estimate or forecast cut-offs for a year that is not in the data.", inputs)
    if round_ is not None and round_ not in rounds:
        return _refuse(f"The sources only contain Round {' and '.join(map(str, rounds))} closing ranks.", inputs)
    year = year or years[-1]
    round_ = round_ or rounds[-1]
    inputs.update(year=year, round=round_)

    notes = [f"Reference: {year} Round {round_} closing ranks (latest final round in the data)."]
    if domicile_state:
        match = next((s for s in states if s.lower() == domicile_state.lower().strip()), None)
        if match is None:
            notes.append(f"No State Quota data for '{domicile_state}'; only All India Quota and Deemed seats are considered.")
        domicile_state = match
    else:
        notes.append("No domicile state given, so State Quota seats are not considered.")
    if category != "GEN":
        notes.append("Deemed University seats are open seats in this dataset, so the GEN closing rank is used for them.")

    by_key = {}
    for r in rows:
        d = r.data
        by_key[(d["year"], d["round"], d["college_code"], d["course"], d["quota"], d["category"])] = r

    options: list[Option] = []
    for r in rows:
        d = r.data
        if d["year"] != year or d["round"] != round_ or d["course"] != course:
            continue
        if colleges and d["college_code"] not in colleges:
            continue
        if quotas and d["quota"] not in quotas:
            continue
        if d["quota"] == "State" and d["state"] != domicile_state:
            continue
        wanted_cat = "GEN" if d["quota"] == "Deemed" else category
        if d["category"] != wanted_cat:
            continue
        ratio = rank / d["closing_rank"]
        band = next((name for name, limit_ in BANDS if ratio <= limit_), None)
        if band is None:
            if not colleges:
                continue
            band = "Outside"
        trend, cites = [], [r.ref]
        for y in years:
            tr = by_key.get((y, round_, d["college_code"], course, d["quota"], wanted_cat))
            if tr:
                trend.append({"year": y, "closing_rank": tr.data["closing_rank"], "ref": tr.ref})
                if tr.ref not in cites:
                    cites.append(tr.ref)
        fee = fees.get((d["college_code"], course))
        if fee:
            cites.append(fee.ref)
        quota_txt = {"AIQ": "All India Quota", "State": f"{d['state']} State Quota", "Deemed": "Deemed University seat"}[d["quota"]]
        if band == "Outside":
            why = (f"Your rank {rank:,} is {round(100 * (ratio - 1))}% beyond the {year} Round {round_} {quota_txt} {wanted_cat} "
                   f"closing rank of {d['closing_rank']:,} [{r.ref}], so this seat closed before your rank in that round.")
        elif band == "Reach":
            why = (f"Your rank {rank:,} is up to 15% beyond the {year} Round {round_} {quota_txt} {wanted_cat} closing rank "
                   f"of {d['closing_rank']:,} [{r.ref}]; it would need the cut-off to move in your favour.")
        else:
            why = (f"Your rank {rank:,} is within the {year} Round {round_} {quota_txt} {wanted_cat} closing rank "
                   f"of {d['closing_rank']:,} [{r.ref}].")
        if len(trend) > 1:
            why += " Closing rank by year: " + ", ".join(f"{t['year']}: {t['closing_rank']:,}" for t in trend) + "."
        if fee:
            why += f" Annual tuition (2025): INR {int(fee.data['annual_tuition_inr']):,} [{fee.ref}]."
        options.append(Option(
            band=band, college=d["college"], college_code=d["college_code"], state=d["state"],
            college_type=d["college_type"], course=course, quota=d["quota"], category_used=wanted_cat,
            closing_rank=d["closing_rank"], reason=why, trend=trend,
            annual_tuition_inr=int(fee.data["annual_tuition_inr"]) if fee else None, citations=cites,
        ))

    order = {name: i for i, (name, _) in enumerate(BANDS + [("Outside", 0)])}
    options.sort(key=lambda o: (order[o.band], o.closing_rank))
    total = len(options)
    options = options[:limit]
    if category == "ST":
        notes.append("Some ST rows are missing in the data; colleges without a recorded ST closing rank are not shown, not estimated.")
    if colleges:
        found = {o.college_code for o in options}
        missing = [c for c in colleges if c not in found]
        if missing:
            notes.append(f"No {year} Round {round_} {category} {course} closing rank in the sources for: {', '.join(missing)} "
                         "(for the quotas you are eligible for). Not estimated.")
        msg = f"Checked AIR {rank:,} ({category}, {course}) against the named college(s)."
        return Prediction(status="ok", message=msg, inputs=inputs, options=options, notes=notes)
    if total == 0:
        msg = (f"No college in the sources had a {year} Round {round_} closing rank at or near AIR {rank:,} for {category} "
               f"({course}). I will not stretch the data to suggest options it does not support.")
    else:
        msg = f"Found {total} option(s) for AIR {rank:,}, {category}, {course}" + (f", showing the first {limit}." if total > limit else ".")
    return Prediction(status="ok", message=msg, inputs=inputs, options=options, notes=notes)
