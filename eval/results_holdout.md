# Evaluation results

Run at 2026-11-15 13:24 · model `qwen2.5:7b-instruct` (ollama) · embeddings `nomic-embed-text` · 14 questions · total 34s

| Metric | Result |
|---|---|
| Overall pass (all checks for a question) | **13/14** (93%) |
| Should-refuse questions correctly refused | 6/6 |
| Answerable questions wrongly refused | 0/8 |
| Expected citation present | 3/3 |
| Median latency, questions that reached the LLM | 2.4s |

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
| H12 | predictor | answer | answered | rag | behaviour:ok content:X route:X | **no** | missing 'not a guarantee' |
| H13 | lookup | answer | answered | rag | behaviour:ok content:ok | yes |  |
| H14 | refuse_missing_row | refuse | refused (model_not_in_sources) | rag | behaviour:ok | yes |  |

Loaded model info: `[{"name": "qwen2.5:7b-instruct", "size_vram_gb": 4.99}, {"name": "nomic-embed-text:latest", "size_vram_gb": 0.34}]`
