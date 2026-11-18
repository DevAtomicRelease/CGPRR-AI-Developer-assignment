"""Build the static slides used between live UI segments. All terminal text is real
output captured on this laptop just before recording (cap/*.txt)."""

import html
import pathlib
import sys

V = pathlib.Path(__file__).parent
REPO = pathlib.Path(sys.argv[1])
P = V / "pages"
P.mkdir(exist_ok=True)

BASE = """<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;width:1280px;height:720px;overflow:hidden;font-family:'Segoe UI',Arial,sans-serif;background:#0f141b;color:#e6e9ef}
.wrap{padding:40px 56px}
h1{font-size:40px;margin:0 0 10px} h2{font-size:26px;margin:0 0 18px;color:#9ec1ff}
.sub{font-size:22px;color:#aab3c2;line-height:1.5}
.term{background:#0a0d12;border:1px solid #2a3240;border-radius:10px;padding:18px 22px;font:15px/1.45 Consolas,monospace;white-space:pre;overflow:hidden}
.p{color:#7ee787}.c{color:#e6e9ef}.dim{color:#8b95a7}.hl{background:#3a3000;color:#ffe08a}
.l{opacity:0;transition:opacity .25s}.l.on{opacity:1}
table{border-collapse:collapse;width:100%;font-size:21px}
td,th{border-bottom:1px solid #2a3240;padding:10px 12px;text-align:left} th{color:#8b95a7;font-weight:600}
.ok{color:#7ee787;font-weight:700}.warn{color:#ffb454;font-weight:700}
.box{margin-top:20px;border-left:4px solid #ffb454;background:#1a1f28;padding:14px 18px;font-size:19px;line-height:1.45}
ol{font-size:22px;line-height:1.7;margin:0;padding-left:28px} li b{color:#9ec1ff}
</style></head><body>__BODY__
<script>
// reveal terminal lines one by one
const ls=[...document.querySelectorAll('.l')];let i=0;const step=+(document.body.dataset.step||60);
function nx(){if(i<ls.length){ls[i++].classList.add('on');setTimeout(nx,step)}} setTimeout(nx,300);
</script></body></html>"""


def page(name, body, step=60):
    (P / name).write_text(BASE.replace("<body>", f'<body data-step="{step}">').replace("__BODY__", body), encoding="utf-8")


def term_lines(cmd_out):
    out = []
    for cmd, text in cmd_out:
        out.append(f'<div class="l"><span class="p">$ </span><span class="c">{html.escape(cmd)}</span></div>')
        for ln in text.rstrip().splitlines():
            out.append(f'<div class="l dim">{html.escape(ln) or "&nbsp;"}</div>')
        out.append('<div class="l">&nbsp;</div>')
    return "".join(out)


cap = V / "cap"
page("title.html", """<div class="wrap" style="padding-top:170px">
<h1>NEET.info private RAG assistant</h1>
<div class="sub">CGPRR AI Developer assignment &middot; Core task (local RAG) + Track A (college predictor)<br>
Open-weight models on one laptop GPU &middot; no cloud API &middot; synthetic data only</div></div>""")

page("local.html", '<div class="wrap"><h2>Everything runs locally</h2><div class="term">' + term_lines([
    ("ollama ps", (cap / "ollama_ps.txt").read_text()),
    ("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv", (cap / "nvidia.txt").read_text()),
]) + "</div></div>", step=180)

# CSV lines around 437 with line numbers, 437 highlighted
rows = (REPO / "data/tables/closing_ranks.csv").read_text(encoding="utf-8").splitlines()
lines = [f'<div class="l dim">   1  {html.escape(rows[0])}</div>', '<div class="l dim">  ...</div>']
for n in range(431, 444):
    cls = "l hl" if n == 437 else "l"
    lines.append(f'<div class="{cls}"><span class="dim">{n:>4}</span>  {html.escape(rows[n - 1])}</div>')
page("csv.html", '<div class="wrap"><h2>data/tables/closing_ranks.csv &mdash; line 437</h2><div class="term" style="font-size:14px">'
     + "".join(lines) + "</div></div>", step=40)

page("checks.html", """<div class="wrap"><h2>How an answer is produced and checked</h2><ol>
<li><b>Injection guard</b> &ndash; "ignore the documents..." is removed and logged</li>
<li><b>Rules</b> &ndash; unknown college, year outside 2023-2025, uncovered topic: refuse before the model</li>
<li><b>Hybrid retrieval</b> &ndash; BM25 + embeddings, filtered by year, round, quota, category</li>
<li><b>Model</b> &ndash; must quote its evidence, then answer with [S#] citations, then set a status</li>
<li><b>Post-checks</b> &ndash; every quote must be verbatim in its source; every number must be in a cited source</li>
<li><b>Otherwise</b> &ndash; the answer is withheld: "Not in the sources"</li></ol></div>""")

ev = (cap / "eval_dev.txt").read_text().splitlines()
ev = [x for x in ev if x.startswith(("PASS", "FAIL")) or x.startswith("| Overall") or x.startswith("| Should")
      or x.startswith("| Answerable") or x.startswith("| Expected")]
page("eval.html", '<div class="wrap" style="padding-top:24px"><h2>python eval/run_eval.py</h2><div class="term" style="font-size:14px">'
     + "".join(f'<div class="l {"" if x.startswith("|") else "dim"}">{html.escape(x.replace("**", ""))}</div>' for x in ev)
     + "</div></div>", step=110)

page("summary.html", """<div class="wrap"><h2>Evaluation results (honest view)</h2>
<table><tr><th>Set</th><th>Questions</th><th>Result</th></tr>
<tr><td>Development set (used while building)</td><td>20</td><td class="ok">20/20</td></tr>
<tr><td>Held-out set, first and only blind run</td><td>14</td><td class="warn">13/14</td></tr>
<tr><td>Held-out set after the fix</td><td>14</td><td class="ok">14/14</td></tr>
<tr><td>Ablation: rule layer off, should-refuse questions</td><td>13</td><td class="warn">11/13 refused</td></tr></table>
<div class="box"><b>Held-out failure H12:</b> "My AIR is 250000 and I'm SC from Bihar, which MBBS colleges are possible?"
&rarr; the model said a closing rank of 96,243 was "higher than" 250,000.<br><b>Fix:</b> any question with the user's own rank
goes to the deterministic predictor.<br><b>Ablation:</b> without rules, the model answered "AIIMS Delhi" with a different Delhi college.</div></div>""")

page("end.html", """<div class="wrap" style="padding-top:150px"><h1>Summary</h1><div class="sub">
Open-weight models on one GPU &middot; cited answers &middot; refusals when sources don't support an answer<br>
Track A predictor that cannot invent cut-offs &middot; evaluation with failures reported<br><br>
<span style="color:#ffb454">Limits:</span> synthetic data, small test set, English only, pattern rules miss some phrasings<br>
<span style="color:#9ec1ff">github.com/DevAtomicRelease/CGPRR-AI-Developer-assignment</span></div></div>""")
print("pages:", sorted(x.name for x in P.iterdir()))
