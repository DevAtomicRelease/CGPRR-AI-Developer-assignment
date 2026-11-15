"""Command-line entry point.

  python -m neetrag build-index
  python -m neetrag ask "What was the 2025 Round 2 AIQ GEN closing rank for MBBS at Ganga Valley Government Medical College?"
  python -m neetrag predict --rank 12000 --category OBC --state "West Bengal"
  python -m neetrag serve --port 8080
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="neetrag", description="NEET.info private RAG assistant (synthetic data demo)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-index", help="embed data/ and write index/")
    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="print the full JSON result")
    p = sub.add_parser("predict", help="list college options for a rank (Track A)")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--category", required=True, help="GEN, EWS, OBC, SC or ST")
    p.add_argument("--course", default="MBBS")
    p.add_argument("--state", default=None, help="domicile state, enables State Quota seats")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--round", type=int, default=None, dest="round_")
    s = sub.add_parser("serve", help="start the web UI / HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    args = ap.parse_args(argv)

    if args.cmd == "build-index":
        from .index import build_index

        print(json.dumps(build_index(), indent=2))
    elif args.cmd == "ask":
        from .assistant import ask

        res = ask(args.question)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print(f"[{res['status']}{' / ' + res['reason'] if res['reason'] else ''}] route={res['route']} "
                  f"flags={res['flags']} latency={res['latency_s']}s\n")
            print(res["answer"])
            if res["citations"]:
                print("\nSources:")
                for c in res["citations"]:
                    print(f"  [{c['id']}] {c['text'][:160]}")
    elif args.cmd == "predict":
        from dataclasses import asdict

        from .predictor import predict

        r = predict(args.rank, args.category, args.course, args.state, args.year, args.round_)
        print(r.message)
        for n in r.notes:
            print("  note:", n)
        for o in r.options:
            print(f"- [{o.band:10}] {o.college} ({o.quota}) closing {o.closing_rank:,}\n    {o.reason}")
        if r.caveat:
            print("\n" + r.caveat)
        if not r.options and r.status == "ok":
            print(json.dumps(asdict(r)["inputs"]))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("neetrag.server:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
