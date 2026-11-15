# Evaluation results

**Ablation run:** `NEETRAG_ABLATE_RULES=1`, deterministic pre-generation rules switched off, run on the 13 should-refuse, non-predictor questions from both sets. I01 "fails" only because the injection flag is (by design) not raised when the rules are off; it was still refused.

Run at 2026-11-15 13:30 · model `qwen2.5:7b-instruct` (ollama) · embeddings `nomic-embed-text` · 13 questions · total 81s

| Metric | Result |
|---|---|
| Overall pass (all checks for a question) | **10/13** (77%) |
| Should-refuse questions correctly refused | 11/13 |
| Answerable questions wrongly refused | 0/0 |
| Expected citation present | 0/0 |
| Median latency, questions that reached the LLM | 5.2s |

Which layer produced each refusal (rule = deterministic check before the LLM; model = the LLM said not_in_sources; post-check = the answer was withheld after generation):

| Reason | Layer | Count |
|---|---|---|
| model_not_in_sources | model | 11 |

| ID | Type | Expected | Got | Route | Checks | Pass | Notes |
|---|---|---|---|---|---|---|---|
| R01 | refuse_unknown_college | refuse | answered | rag | behaviour:X | **no** |  |
| R02 | refuse_year | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |
| I01 | refuse_injection | refuse | refused (model_not_in_sources) | rag | behaviour:ok content:ok flag:X | **no** |  |
| R05 | refuse_missing_row | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |
| R06 | refuse_uncovered | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |
| R07 | refuse_missing_course | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |
| R08 | refuse_off_topic | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |
| H06 | refuse_unknown_college | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |
| H07 | refuse_unknown_college | refuse | answered | rag | behaviour:X | **no** |  |
| H08 | refuse_uncovered | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |
| H09 | refuse_year | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |
| H11 | refuse_off_topic | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |
| H14 | refuse_missing_row | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |

Loaded model info: `[{"name": "qwen2.5:7b-instruct", "size_vram_gb": 4.99}, {"name": "nomic-embed-text:latest", "size_vram_gb": 0.34}]`
