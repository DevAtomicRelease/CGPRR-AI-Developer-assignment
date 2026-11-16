# Evaluation results

Run at 2026-11-16 15:45 · model `qwen2.5:7b-instruct` (ollama) · embeddings `nomic-embed-text` · 14 questions · total 43s

| Metric | Result |
|---|---|
| Overall pass (all checks for a question) | **14/14** (100%) |
| Should-refuse questions correctly refused | 6/6 |
| Answerable questions wrongly refused | 0/8 |
| Expected citation present | 3/3 |
| Median latency, questions that reached the LLM | 2.7s |

Which layer produced each refusal (rule = deterministic check before the LLM; model = the LLM said not_in_sources; post-check = the answer was withheld after generation):

| Reason | Layer | Count |
|---|---|---|
| model_not_in_sources | model | 2 |
| uncovered_topic | rule | 1 |
| unknown_college | rule | 2 |
| year_not_covered | rule | 1 |

| ID | Type | Expected | Got | Route | Checks | Pass | Notes |
|---|---|---|---|---|---|---|---|
| H01 | lookup | answer | answered | rag | behaviour:ok content:ok citation:ok | yes |  |
| H02 | lookup | answer | answered | rag | behaviour:ok content:ok citation:ok | yes |  |
| H03 | process | answer | answered | rag | behaviour:ok content:ok | yes |  |
| H04 | lookup | answer | answered | rag | behaviour:ok content:ok | yes |  |
| H05 | process | answer | answered | rag | behaviour:ok content:ok | yes |  |
| H06 | refuse_unknown_college | refuse | refused (unknown_college) | rag | behaviour:ok | yes |  |
| H07 | refuse_unknown_college | refuse | refused (unknown_college) | rag | behaviour:ok | yes |  |
| H08 | refuse_uncovered | refuse | refused (uncovered_topic) | rag | behaviour:ok | yes |  |
| H09 | refuse_year | refuse | refused (year_not_covered) | rag | behaviour:ok | yes |  |
| H10 | planted_figure | answer | answered | rag | behaviour:ok content:ok citation:ok | yes |  |
| H11 | refuse_off_topic | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |
| H12 | predictor | answer | answered | predictor | behaviour:ok content:ok route:ok | yes |  |
| H13 | lookup | answer | answered | rag | behaviour:ok content:ok | yes |  |
| H14 | refuse_missing_row | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |

Loaded model info: `[{"name": "qwen2.5:7b-instruct", "size_vram_gb": 4.99}, {"name": "nomic-embed-text:latest", "size_vram_gb": 0.34}]`
