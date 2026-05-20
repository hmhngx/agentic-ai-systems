# 08 — Evaluation

## Objective

This module covers systematic evaluation of RAG and agent systems—RAGAS metrics, LLM-as-a-judge patterns, LangSmith tracing, and hallucination rate measurement. After completing this module, you will be able to build evaluation datasets, run RAGAS faithfulness and relevancy benchmarks, implement custom LLM judges for domain-specific quality criteria, instrument pipelines with LangSmith traces, and report hallucination rates with statistical confidence intervals.

## Core concepts

| Concept | Mental model (one sentence) | Why it matters in production |
|---------|----------------------------|------------------------------|
| RAGAS faithfulness | Measures whether the generated answer is grounded in the retrieved context | Detects hallucination—the single most important RAG quality metric for user trust |
| RAGAS context precision | Measures whether retrieved chunks are relevant to the query | Diagnoses retrieval failures separately from generation failures |
| RAGAS context recall | Measures whether retrieved context contains the information needed to answer | Identifies retrieval gaps before blaming the LLM for wrong answers |
| LLM-as-a-judge | Using a stronger model to score outputs on rubrics too nuanced for programmatic checks | Captures quality dimensions (tone, completeness, reasoning) that metrics alone miss |
| LangSmith tracing | Recording every LLM call, retrieval step, and tool invocation with latency and token counts | Makes production failures reproducible; enables regression detection across deployments |

## Implementation artifacts

| File | Description | Status |
|------|-------------|--------|
| `ragas_eval.py` | RAGAS evaluation runner with faithfulness, answer relevancy, context precision, and context recall | 🔲 Pending |
| `llm_judge.py` | Custom LLM-as-a-judge with rubric-based scoring for domain-specific quality criteria | 🔲 Pending |
| `langsmith_tracing.py` | LangSmith instrumentation wrapper for tracing full RAG and agent pipelines | 🔲 Pending |
| `hallucination_tracker.py` | Faithfulness-based hallucination rate measurement with confidence intervals | 🔲 Pending |
| `eval_dataset.json` | Curated evaluation dataset with queries, ground-truth answers, and relevance judgments | 🔲 Pending |

## Key decisions & tradeoffs

- RAGAS will be the primary automated evaluation framework; custom LLM judges will supplement for dimensions RAGAS does not cover (citation accuracy, tone, completeness).
- Evaluation datasets will contain a minimum of 50 query-answer pairs per module, manually verified for ground-truth quality—synthetic-only datasets will not be used for final scoring.
- LangSmith tracing will be enabled in all development and staging environments; production tracing will sample at 10% to control cost. Set `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` from the root `.env.example`.
- Hallucination rate will be reported as the complement of RAGAS faithfulness, with 95% confidence intervals across the evaluation set.
- Evaluation will run on every PR that touches retrieval or generation logic; score regressions greater than 0.05 on any metric will block merge.

## Evaluation

| Metric | Target | Actual |
|--------|--------|--------|
| RAGAS faithfulness (full pipeline) | ≥ 0.85 | |
| RAGAS answer relevancy | ≥ 0.80 | |
| RAGAS context precision | ≥ 0.80 | |
| RAGAS context recall | ≥ 0.75 | |
| Hallucination rate (1 - faithfulness) | ≤ 5% | |

## References

- Es, S., et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217. https://arxiv.org/abs/2309.15217
- RAGAS documentation: https://docs.ragas.io/
- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023. https://arxiv.org/abs/2306.05685
- LangSmith documentation: https://docs.smith.langchain.com/
- LangChain Evaluation guide: https://python.langchain.com/docs/guides/evaluation/
