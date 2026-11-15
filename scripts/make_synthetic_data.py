"""Generate the synthetic NEET-UG counselling dataset shipped in data/.

Every college name, rank, seat count and fee produced here is FICTIONAL.
The table layout mirrors the style of public counselling documents (closing
rank by year / round / quota / category, seat matrix, fee schedule) so the
assistant can be demonstrated without using any real student or college data.

The generator is deterministic (fixed seed), so re-running it reproduces the
exact files committed in the repository.

Usage:  python scripts/make_synthetic_data.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 20261114
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
DOCS = ROOT / "data" / "docs"

YEARS = [2023, 2024, 2025]
ROUNDS = [1, 2]
CATEGORIES = ["GEN", "EWS", "OBC", "SC", "ST"]
CATEGORY_FACTOR = {"GEN": 1.0, "EWS": 1.12, "OBC": 1.18, "SC": 4.8, "ST": 6.5}
# Central-style reservation shares used to split seats.
CATEGORY_SHARE = {"EWS": 0.10, "OBC": 0.27, "SC": 0.15, "ST": 0.075}
YEAR_FACTOR = {2023: 1.00, 2024: 1.07, 2025: 0.97}

# (code, name, city, state, type, established, mbbs_seats, base_closing_rank, bds_base or None)
COLLEGES = [
    ("HRGMC", "Hooghly Riverside Government Medical College", "Kolkata", "West Bengal", "Government", 1961, 250, 900, 31000),
    ("SGMC", "Sundarban Government Medical College", "Diamond Harbour", "West Bengal", "Government", 2019, 100, 6500, None),
    ("NBHGMC", "North Bengal Hills Government Medical College", "Siliguri", "West Bengal", "Government", 1968, 200, 9000, None),
    ("KCGMC", "Kalinga Coast Government Medical College", "Cuttack", "Odisha", "Government", 1944, 250, 4200, 38000),
    ("MGMC", "Mahanadi Government Medical College", "Sambalpur", "Odisha", "Government", 1959, 150, 11500, None),
    ("GVGMC", "Ganga Valley Government Medical College", "Patna", "Bihar", "Government", 1925, 200, 3800, 36000),
    ("MIGMC", "Mithila Government Medical College", "Darbhanga", "Bihar", "Government", 1946, 120, 13000, None),
    ("CGMC", "Chotanagpur Government Medical College", "Ranchi", "Jharkhand", "Government", 1960, 180, 10200, None),
    ("BGMC", "Brahmaputra Government Medical College", "Guwahati", "Assam", "Government", 1960, 200, 7400, None),
    ("YIMS", "Yamuna Institute of Medical Sciences", "New Delhi", "Delhi", "Government", 1956, 150, 250, None),
    ("DPGMC", "Deccan Plateau Government Medical College", "Pune", "Maharashtra", "Government", 1964, 200, 1800, 33000),
    ("KOGMC", "Konkan Government Medical College", "Ratnagiri", "Maharashtra", "Government", 2021, 100, 14500, None),
    ("KAGMC", "Kaveri Government Medical College", "Mysuru", "Karnataka", "Government", 1924, 150, 5200, None),
    ("COGMC", "Coromandel Government Medical College", "Chennai", "Tamil Nadu", "Government", 1950, 250, 1300, 30000),
    ("AGMC", "Awadh Government Medical College", "Lucknow", "Uttar Pradesh", "Government", 1911, 250, 2600, None),
    ("BUGMC", "Bundelkhand Government Medical College", "Jhansi", "Uttar Pradesh", "Government", 1968, 150, 16000, None),
    ("SLIMS", "Salt Lake Institute of Medical Sciences", "Kolkata", "West Bengal", "Private", 2008, 150, 52000, None),
    ("EHMC", "Eastern Health Medical College", "Bhubaneswar", "Odisha", "Private", 2011, 150, 61000, None),
    ("NHMC", "Nandi Hills Medical College", "Bengaluru", "Karnataka", "Private", 2005, 150, 48000, None),
    ("MPMC", "Marina Private Medical College", "Chennai", "Tamil Nadu", "Private", 2003, 150, 55000, None),
    ("VDUMC", "Vindhya Deemed University Medical College", "Indore", "Madhya Pradesh", "Deemed", 2002, 150, 125000, None),
    ("SDUMC", "Sahyadri Deemed University Medical College", "Navi Mumbai", "Maharashtra", "Deemed", 1998, 150, 98000, 210000),
]

FEES = {  # annual tuition (INR) by college type and course; hostel per year
    ("Government", "MBBS"): (15000, 65000),
    ("Government", "BDS"): (12000, 50000),
    ("Private", "MBBS"): (850000, 1450000),
    ("Deemed", "MBBS"): (1800000, 2500000),
    ("Deemed", "BDS"): (650000, 900000),
}


def quotas_for(ctype: str) -> list[str]:
    if ctype == "Government":
        return ["AIQ", "State"]
    if ctype == "Private":
        return ["State"]
    return ["Deemed"]


def categories_for(quota: str) -> list[str]:
    # Deemed-university seats are open seats in this synthetic dataset.
    return ["GEN"] if quota == "Deemed" else CATEGORIES


def split_seats(total: int, cats: list[str]) -> dict[str, int]:
    if cats == ["GEN"]:
        return {"GEN": total}
    seats = {c: int(round(total * CATEGORY_SHARE[c])) for c in cats if c != "GEN"}
    seats["GEN"] = total - sum(seats.values())
    return seats


def main() -> None:
    rng = random.Random(SEED)
    TABLES.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    cutoff_rows: list[dict] = []
    seat_rows: list[dict] = []
    fee_rows: list[dict] = []

    for code, name, city, state, ctype, _est, mbbs_seats, base, bds_base in COLLEGES:
        courses = [("MBBS", mbbs_seats, base)]
        if bds_base:
            courses.append(("BDS", 50 if ctype == "Government" else 100, bds_base))

        for course, total_seats, course_base in courses:
            lo, hi = FEES[(ctype, course)]
            tuition = int(round(rng.uniform(lo, hi), -3))
            hostel = int(round(rng.uniform(30000, 90000) if ctype != "Government" else rng.uniform(6000, 24000), -3))
            fee_rows.append({
                "year": 2025, "college_code": code, "college": name, "state": state,
                "college_type": ctype, "course": course,
                "annual_tuition_inr": tuition, "annual_hostel_inr": hostel,
            })

            for quota in quotas_for(ctype):
                if quota == "AIQ":
                    quota_seats = int(round(total_seats * 0.15))
                    q_factor = 1.0
                elif quota == "State" and ctype == "Government":
                    quota_seats = total_seats - int(round(total_seats * 0.15))
                    q_factor = 1.6
                else:
                    quota_seats = total_seats
                    q_factor = 1.0
                cats = categories_for(quota)

                for year in YEARS:
                    year_seats = quota_seats + (10 if (year == 2025 and ctype == "Government" and quota == "State" and course == "MBBS") else 0)
                    for cat, n in split_seats(year_seats, cats).items():
                        seat_rows.append({
                            "year": year, "college_code": code, "college": name, "state": state,
                            "college_type": ctype, "course": course, "quota": quota,
                            "category": cat, "seats": n,
                        })

                    for cat in cats:
                        # Leave genuine gaps: some category/quota combinations were "not allotted"
                        # in a given year, so the honest answer for them is "not in the sources".
                        if cat == "ST" and rng.random() < 0.12:
                            continue
                        closing_r1 = course_base * q_factor * CATEGORY_FACTOR[cat] * YEAR_FACTOR[year]
                        closing_r1 *= rng.uniform(0.94, 1.06)
                        closing_r1 = max(5, int(closing_r1))
                        for rnd in ROUNDS:
                            closing = closing_r1 if rnd == 1 else int(closing_r1 * rng.uniform(1.04, 1.12))
                            opening = max(1, int(closing * rng.uniform(0.35, 0.6)))
                            cutoff_rows.append({
                                "year": year, "round": rnd, "college_code": code, "college": name,
                                "city": city, "state": state, "college_type": ctype, "course": course,
                                "quota": quota, "category": cat,
                                "opening_rank": opening, "closing_rank": closing,
                            })

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write(TABLES / "closing_ranks.csv", cutoff_rows)
    write(TABLES / "seat_matrix.csv", seat_rows)
    write(TABLES / "fees_2025.csv", fee_rows)

    # College profiles document (derived from the same synthetic facts).
    lines = [
        "# College profiles (SYNTHETIC)",
        "",
        "All colleges below are fictional and exist only in this demo dataset.",
        "Seat counts are total sanctioned seats for 2025; fees are in fees_2025.csv.",
        "",
    ]
    for code, name, city, state, ctype, est, mbbs_seats, _base, bds_base in COLLEGES:
        courses = "MBBS and BDS" if bds_base else "MBBS"
        quotas = ", ".join({"AIQ": "All India Quota (15%)", "State": "State Quota", "Deemed": "Deemed University (open, all-India)"}[q] for q in quotas_for(ctype))
        lines += [
            f"## {name} ({code})",
            "",
            f"{name} is a {ctype.lower()} medical college in {city}, {state}, established in {est}. "
            f"It offers {courses}. It has {mbbs_seats} MBBS seats"
            + (f" and {50 if ctype == 'Government' else 100} BDS seats" if bds_base else "")
            + f". Seats are filled through: {quotas}.",
            "",
        ]
    (DOCS / "college_profiles.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"closing_ranks.csv: {len(cutoff_rows)} rows")
    print(f"seat_matrix.csv:   {len(seat_rows)} rows")
    print(f"fees_2025.csv:     {len(fee_rows)} rows")


if __name__ == "__main__":
    main()
