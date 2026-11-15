# Design note: NEET.info private RAG assistant

*CGPRR AI Developer assignment. Core task (local RAG) plus Track A (college predictor), both for NEET.info.*

## 1. Architecture

```
question
  ├─ 1. injection guard ── matched sentence removed, event logged (logs/security.jsonl)
  ├─ 2. entity rules ───── college, year, round, category, quota, course, state, user's rank
  ├─ 3. router
  │     ├─ user gave a rank ──► Track A predictor (pure Python over CSV, no LLM) ──► cited options
  │     └─ otherwise ─────────► coverage rules: unknown college / year outside data / uncovered topic ──► refuse
  ├─ 4. hybrid retrieval ───── BM25 + nomic-embed-text, reciprocal-rank fusion, metadata filters
  │                            best dense score < 0.60 ──► refuse
  ├─ 5. Qwen2.5-7B (Ollama) ── JSON: evidence quotes → answer with [S#] → status
  └─ 6. post-checks ────────── quotes must be verbatim in the named source; every number in the
                               answer must be in a cited source; otherwise withhold ──► cited answer
```

**Chunks.** Each CSV row is one chunk and each Markdown `##` section is one chunk (2,135 in all).
Citation ids point at something a reviewer can open: `closing_ranks.csv#L437` is line 437 of the
file. Table rows carry metadata (year, round, college, quota, category, course), so "2025 Round 2
AIQ GEN at Ganga Valley" narrows retrieval to the right rows before ranking. The documents always
stay in the candidate set.

**Why the extra layers around the model.** A 7B model is good at reading a quoted record and bad
at two things that matter most for counselling: comparing numbers, and noticing that the college
asked about is not the one it retrieved. Both failures showed up in testing (section 4). The design
therefore keeps decisions that can be made by code in code. The predictor is deterministic. The
"not in the sources" decisions for unknown colleges, years and topics are rules. The model is only
asked to read the sources and phrase an answer, and its output is checked again afterwards.

**Track A.** Given a rank, category, course and optional domicile state, the predictor takes the
latest final-round closing ranks (2025 Round 2) for every quota the candidate can use. It marks each
college Likely (rank ≤ 85% of the closing rank), Borderline (≤ 100%) or Reach (≤ 115%). Each option
shows the closing-rank trend for 2023-2025 and the tuition fee, with the CSV line for every figure.
It refuses requests that would need an invented cut-off: a year outside the data (2026), a category
or quota the data lacks (NRI, PwD), or a course the data lacks. Missing rows (for example some ST
rows) are reported as missing, not estimated. Every result carries a "not a guarantee" caveat.

## 2. Model choice

| Role | Model | Why |
|---|---|---|
| Answer generation | **Qwen2.5-7B-Instruct**, Q4_K_M GGUF via Ollama | Apache-2.0 open weights; reliable instruction following and JSON output at 7B; fits the 8 GB development GPU with an 8k context. |
| Embeddings | **nomic-embed-text v1** (137M) | Apache-2.0, 768-dim, runs on the same Ollama server, 0.3 GB VRAM. |
| Office upgrade | **Qwen2.5-32B-Instruct** (AWQ 4-bit via vLLM, or Q4 via Ollama) | Same family and prompt format; noticeably better at reasoning over tables; fits 48 GB with room for concurrency. |

Ollama's grammar-constrained JSON output (`format` = JSON schema) makes the reply parseable every
time. The schema puts `evidence` first and requires at least one quote, so the model must look at
the sources before it decides. An earlier schema that asked for the status first, with no required
quote, refused 5 of the 12 answerable development questions. Temperature is 0 and the seed is fixed, so runs are repeatable. The code also speaks
the OpenAI-compatible API, so vLLM or llama.cpp can replace Ollama without code changes.

## 3. GPU and memory estimate

**Measured on the development laptop** (RTX 4060 Laptop, 8 GB, `scripts/bench.py`, median of 5):
the chat model holds 4.65 GiB and the embedder 0.31 GiB. A typical prompt (~1,200 tokens of
sources) is processed at about 3,000 tok/s, and the answer (~125 tokens) is generated at 44 tok/s.
That is about 3.2 s of generation per answer, and 3.3 s median end to end in the evaluation. Building
the index embeds 2,135 chunks in 177 s.

**Estimate for one RTX 6000-class card. Assumed: RTX 6000 Ada, 48 GB GDDR6, ~960 GB/s, 300 W.**
Token generation is limited by memory bandwidth, so speed scales roughly with bandwidth ÷ model size.

| Setup on the RTX 6000 | VRAM | Single-user speed (estimate) |
|---|---|---|
| Qwen2.5-7B Q4 (as tested) + embedder | ~5.5 GB with 8k context | ~130-160 tok/s, about 1 s per answer |
| Qwen2.5-7B FP16 | ~15 GB weights + ~0.5 GB KV per 8k-token request | ~50-60 tok/s |
| **Qwen2.5-32B AWQ (recommended)** | ~19 GB weights + KV ~0.26 MB/token (64 layers × 8 KV heads × 128 × 2 × 2 bytes): 32k cached tokens ≈ 8.5 GB → **~30 GB total** | ~35-45 tok/s single stream, about 4-5 s per answer; vLLM batching gives several hundred tok/s across ~15 users |

That leaves about 15 GB of headroom on a 48 GB card. On the newer 96 GB Blackwell RTX PRO 6000, the
32B model could run in FP8 instead, or several business instances could run side by side.

## 4. Evaluation method and results

Each question's expected behaviour (answer or refuse), required strings, forbidden strings,
expected citation and expected route are written in a JSONL file before the run. `eval/run_eval.py`
checks them automatically, and for predictor questions it re-reads every cited CSV line to confirm
the closing rank and band.

| Set | Pass | Notes |
|---|---|---|
| Development, 20 questions (8 should refuse) | 20/20 | The system was tuned on this set, so the score is optimistic. |
| **Held-out, 14 questions, first blind run** | **13/14** | Failure H12: "My AIR is 250000 and I'm SC from Bihar, which MBBS colleges are possible?" missed the predictor route, and the RAG model wrote that a closing rank of 96,243 was "higher than" 250,000. The numbers were real, so the number check passed, but the advice was wrong. Fix: any question containing the user's own rank now goes to the predictor. It then passed 14/14. |
| Ablation: 13 should-refuse questions, rule layer off | 11/13 refused | The two misses were entity substitution: asked about AIIMS Delhi and KGMU Lucknow, the model answered with the nearest indexed colleges (YIMS Delhi, AGMC Lucknow), citing real rows. |

What this shows: the refusal guarantee comes mainly from the rule layer and the deterministic
predictor. On the development set only 2 of 8 refusals came from the model itself. The model is
reliable at quoting and citing a retrieved record, but it should not be trusted on its own to judge
that a figure is missing. Limits: 34 questions on synthetic data is a smoke test, not a benchmark.
String checks can miss a correct paraphrase. The rules will miss phrasings they were not written for.

## 5. Data privacy

- **Nothing leaves the machine.** The app calls only the local model server (127.0.0.1:11434).
  The only network use is the one-time model download. The web UI binds to 127.0.0.1, and Docker
  publishes only 127.0.0.1:8080. No cloud GPU or API was used for development or testing.
- **Where things live:** source data in `data/`; the index in `index/` (`chunks.jsonl`,
  `embeddings.npy`); model weights in the Ollama store (`~/.ollama` or a Docker volume); logs as
  append-only JSON lines in `logs/queries.jsonl` and `logs/security.jsonl`.
- **Logs** currently keep the question text, which is fine for synthetic demo data. With real users
  I would hash or redact identifiers, keep logs for a fixed period (for example 30 days), and keep
  security events separate for audit.
- **Keeping businesses apart in the office.** Run one app instance per business (NEET.info, Hirrx,
  CGPRR editorial, Smart Catalyst), each with its own data, index and log directories, its own OS
  user and file permissions, and its own container. The GPU model server is shared but stateless:
  it receives a prompt and returns text, and it stores no documents. So sharing the card does not
  share data. NEET.info and Smart Catalyst indexes stay separate, so student records never move
  between them automatically. On the workstation: keep Ollama bound to localhost
  (`OLLAMA_HOST=127.0.0.1`), block outbound traffic after models are downloaded, pin model files by
  checksum, and put the UI behind the office login.

## 6. Cost

- **Development:** run entirely on an existing laptop. The GPU draws about 85-90 W during
  inference, so one answer (~3.3 s) uses about 0.08 Wh. That is roughly 12,000 answers per kWh, well
  under ₹0.01 per answer at an assumed ₹8-10 per kWh. The whole evaluation (34 questions) costs a
  fraction of a paisa in electricity.
- **Office workstation (estimate, not a quote):** an RTX 6000 Ada-class workstation draws about
  500 W under load (300 W GPU + system). At 8 hours a day that is about 4 kWh/day, or about
  ₹1,000-1,200 a month at ₹8-10/kWh. The card itself is in the range of several lakh rupees.
  Spread over 3 years, hardware dominates the running cost, not electricity. There is no per-query
  API fee, and the same card can serve the other businesses' assistants.
