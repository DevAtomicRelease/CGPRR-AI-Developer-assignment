"""Run the evaluation set and write a score table.

  python eval/run_eval.py                # all questions
  python eval/run_eval.py --ids A01 R03  # a subset

Writes eval/results.json (full outputs) and eval/results.md (score table).
Each question's expected behaviour is fixed in eval/questions.jsonl before running.

Checks per question (a check that does not apply is skipped, not counted as a pass):
  behaviour  answered vs refused matches "expect"
  content    every "must_contain" string (or one of "must_contain_any") is in the answer;
             no "must_not_contain" string is in the answer (comparison ignores commas and case)
  citation   every id in "must_cite" is among the returned citations
  route/flag the expected route (rag/predictor) and safety flag were used
  predictor  every listed option is backed by the cited CSV row and its band is consistent
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neetrag.assistant import ask  # noqa: E402
from neetrag.config import SETTINGS  # noqa: E402
from neetrag.llm import server_info  # noqa: E402
from neetrag.predictor import BANDS  # noqa: E402

LAYER = {"model_not_in_sources": "model", "unverified_numbers": "post-check", "no_citation": "post-check",
         "llm_error": "error", "malformed_output": "post-check", "predictor_refused": "rule (predictor)", "missing_category": "rule (predictor)"}


def norm(s: str) -> str:
    return s.replace(",", "").lower()


def csv_line(ref: str) -> dict:
    name, line = ref.split("#L")
    path = SETTINGS.data_dir / "tables" / name
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[int(line) - 2]


def check_predictor(res: dict) -> tuple[bool, str]:
    p = res.get("prediction") or {}
    opts = p.get("options") or []
    if not opts:
        return False, "no options"
    rank = p["inputs"]["rank"]
    for o in opts:
        row = csv_line(o["citations"][0])
        if int(row["closing_rank"]) != o["closing_rank"] or row["college_code"] != o["college_code"]:
            return False, f"{o['college_code']}: cited row does not match"
        if (int(row["year"]), int(row["round"])) != (p["inputs"]["year"], p["inputs"]["round"]):
            return False, f"{o['college_code']}: cited row is from another year or round"
        ratio = rank / int(row["closing_rank"])
        # "Outside" is only used when the user named a college the rank is beyond.
        expected = next((name for name, limit in BANDS if ratio <= limit), "Outside")
        if expected != o["band"]:
            return False, f"{o['college_code']}: band {o['band']} inconsistent (ratio {ratio:.2f})"
        if o["quota"] == "State" and o["state"] != p["inputs"]["domicile_state"]:
            return False, f"{o['college_code']}: state quota outside domicile"
    return True, f"{len(opts)} options verified against CSV"


def grade(q: dict, res: dict) -> dict:
    answer = norm(res["answer"])
    refused = res["status"] in {"refused", "needs_input"}
    checks: dict[str, bool] = {}
    notes: list[str] = []

    checks["behaviour"] = refused == (q["expect"] == "refuse")
    content = True
    for s in q.get("must_contain", []):
        if norm(s) not in answer:
            content = False
            notes.append(f"missing '{s}'")
    if q.get("must_contain_any") and not any(norm(s) in answer for s in q["must_contain_any"]):
        content = False
        notes.append(f"none of {q['must_contain_any']}")
    for s in q.get("must_not_contain", []):
        if norm(s) in answer:
            content = False
            notes.append(f"contains forbidden '{s}'")
    if any(k in q for k in ("must_contain", "must_contain_any", "must_not_contain")):
        checks["content"] = content
    if q.get("must_cite"):
        got = {c["id"] for c in res["citations"]}
        missing = [c for c in q["must_cite"] if c not in got]
        checks["citation"] = not missing
        if missing:
            notes.append(f"not cited {missing}")
    if q.get("expect_route"):
        checks["route"] = res["route"] == q["expect_route"]
    if q.get("expect_flag"):
        checks["flag"] = q["expect_flag"] in res["flags"]
    if q.get("check_predictor"):
        ok, why = check_predictor(res)
        checks["predictor"] = ok
        notes.append(why)
    if not refused and q["expect"] == "answer" and res["route"] == "rag" and not res["citations"]:
        checks["citation"] = False
        notes.append("answer without citations")
    return {"checks": checks, "pass": all(checks.values()), "notes": notes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--file", default=str(ROOT / "eval" / "questions.jsonl"))
    args = ap.parse_args()
    qs = [json.loads(ln) for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.ids:
        qs = [q for q in qs if q["id"] in args.ids]

    ask("warm-up: what does closing rank mean?")  # load models so latency numbers are comparable
    rows = []
    t_all = time.time()
    for q in qs:
        res = ask(q["question"])
        g = grade(q, res)
        rows.append({"q": q, "res": res, "grade": g})
        mark = "PASS" if g["pass"] else "FAIL"
        print(f"{mark} {q['id']:4} {res['status']:9} {res['route']:9} {res['latency_s']:>6}s  {'; '.join(g['notes'])}")

    total = len(rows)
    passed = sum(r["grade"]["pass"] for r in rows)
    should_answer = [r for r in rows if r["q"]["expect"] == "answer"]
    should_refuse = [r for r in rows if r["q"]["expect"] == "refuse"]
    correct_refusals = sum(r["res"]["status"] != "answered" for r in should_refuse)
    false_refusals = sum(r["res"]["status"] != "answered" for r in should_answer)
    cite_rows = [r for r in rows if "citation" in r["grade"]["checks"]]
    cite_ok = sum(r["grade"]["checks"]["citation"] for r in cite_rows)
    rag_lat = [r["res"]["latency_s"] for r in rows if r["res"]["route"] == "rag" and r["res"]["reason"] in ("", "model_not_in_sources")]
    info = server_info()

    md = [
        "# Evaluation results",
        "",
        f"Run at {time.strftime('%Y-%m-%d %H:%M')} · model `{SETTINGS.chat_model}` ({SETTINGS.llm_backend}) · "
        f"embeddings `{SETTINGS.embed_model}` · {total} questions · total {time.time() - t_all:.0f}s",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Overall pass (all checks for a question) | **{passed}/{total}** ({100 * passed / total:.0f}%) |",
        f"| Should-refuse questions correctly refused | {correct_refusals}/{len(should_refuse)} |",
        f"| Answerable questions wrongly refused | {false_refusals}/{len(should_answer)} |",
        f"| Expected citation present | {cite_ok}/{len(cite_rows)} |",
        f"| Median latency, questions that reached the LLM | {statistics.median(rag_lat):.1f}s |" if rag_lat else "| Median latency | n/a |",
        "",
        "Which layer produced each refusal (rule = deterministic check before the LLM; "
        "model = the LLM said not_in_sources; post-check = the answer was withheld after generation):",
        "",
        "| Reason | Layer | Count |",
        "|---|---|---|",
        *[f"| {reason} | {LAYER.get(reason, 'rule')} | {n} |" for reason, n in sorted(
            Counter(r["res"]["reason"] for r in rows if r["res"]["status"] != "answered").items())],
        "",
        "| ID | Type | Expected | Got | Route | Checks | Pass | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        q, res, g = r["q"], r["res"], r["grade"]
        checks = " ".join(f"{k}:{'ok' if v else 'X'}" for k, v in g["checks"].items())
        status = res["status"] + (f" ({res['reason']})" if res["reason"] else "")
        md.append(f"| {q['id']} | {q['type']} | {q['expect']} | {status} | {res['route']} | {checks} | "
                  f"{'yes' if g['pass'] else '**no**'} | {'; '.join(g['notes']).replace('|', '/')} |")
    md += ["", f"Loaded model info: `{json.dumps(info.get('loaded', ''))}`", ""]
    out_md = ROOT / "eval" / "results.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    (ROOT / "eval" / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "\n".join(md[4:11]))
    print(f"\nWrote {out_md.relative_to(ROOT)} and eval/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
