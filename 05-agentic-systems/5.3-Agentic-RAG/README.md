# Day 11 — Agentic RAG (Week 2 Review)

A LangGraph agent that uses **RAG as a tool it decides whether to call**, then
**self-corrects** when its answer isn't grounded — gated on a RAGAS faithfulness
score. Offline & deterministic by default; `USE_LLM=1` / `USE_RAGAS=1` swap in
real OpenRouter generation and the real `ragas` library.

## Architecture

```
START -> decide --(retrieve)--> retrieve_and_generate -> evaluate
                                      ^                       |
                              refine  |        route_after_eval
                                      |         |       |        |
                                      +--<------(refine)|        (finalize)--> finalize -> END
                                                        (fallback)----------> fallback -> END
        decide --(direct)--> direct_answer ------------------------------------------> END
```

- **decide** — SHOULD I retrieve? (factual KB question) vs answer **direct**
  (self-contained premise / meta question).
- **retrieve_and_generate** — calls the `rag_v1_complete` tool (tf-idf retrieval
  in Qdrant `:memory:` → grounded, cited answer).
- **evaluate** — RAGAS faithfulness of the answer vs its retrieved context, plus a
  root **failure-mode** diagnosis.
- **refine** (Reflexion) — rewrites the query from accumulated failure diagnoses.
- **fallback** — graceful abstention that reports the **best-scoring** attempt.

The `refine → retrieve_and_generate` back-edge makes this a directed *cyclic*
graph; the `attempt <= max_retries` guard keeps the cycle finite.

## Two thresholds

- `0.70` — reflexion trigger / serve gate (re-query if faithfulness < 0.70).
- `0.75` — the **mean** faithfulness over served, retrieve-routed answers on the
  eval dataset (the headline goal).

## The deterministic reflexion mechanism

The corpus uses *formal* vocabulary (`retention`, `purged`, `quota`); some
questions use *casual* vocabulary (`keep`, `deleting`). With `idf = log(N/df)`,
corpus-ubiquitous terms (`helios`) contribute 0, so a casually-worded question
grounds below `GROUND_MIN`, the generator emits an unsupported guess, faithfulness
drops, and `refine` expands casual→formal — sharpening retrieval onto the owning
document. Vocabulary mismatch is *the* classic retrieval failure; query expansion
is the classic fix. Fully deterministic, zero keys.

The offline generator deliberately **simulates** the canonical RAG failure: when
retrieval grounds weakly it answers from the *question's own words* (absent from
the chunks), which the faithfulness proxy then flags — exactly as it would flag a
real model fabricating from parametric memory under `USE_LLM=1`.

## Run

```bash
python -m venv .venv && . .venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python agentic_rag.py            # offline demo: 4 paths + eval table
python agentic_rag.py --eval     # full-pipeline RAGAS eval table
python -m pytest -v              # 68 machine-checked assertions
```

Optional real backends: copy `.env.example` to `.env`, set `OPENROUTER_API_KEY`
and `USE_LLM=1` (real generation/decision/refine) and/or `USE_RAGAS=1` (real
RAGAS). Topology is identical.

## Goal → evidence

| Goal / task | Where | Test |
|---|---|---|
| Wire `rag_v1_complete` as a tool | `src/rag_tool.py` | `tests/test_rag_tool.py` |
| Agent decides retrieve vs answer-from-context | `src/decision.py`, `decide_node` | `tests/test_decision.py`, `test_concepts.py` |
| Reflexion: refine when faithfulness < 0.70 | `src/reflexion.py`, `route_after_eval` | `tests/test_reflexion.py`, `test_graph.py` |
| Max 2 retries, then graceful fallback | `route_after_eval`, `fallback_node` | `tests/test_graph.py`, `test_concepts.py` |
| Run RAGAS eval on the full pipeline | `src/ragas_eval.py` | `tests/test_ragas_eval.py` |
| Mean faithfulness ≥ 0.75 | `eval_dataset.json` | `tests/test_ragas_eval.py` |
| Read a RAGAS table → what to fix | `src/remediation.py` | `tests/test_concepts.py` |
| Trace hallucination → root failure mode | `src/faithfulness.py` | `tests/test_faithfulness.py`, `test_concepts.py` |

## Failure modes (for tracing a hallucination)

| Mode | Meaning | Fix |
|---|---|---|
| `NO_RETRIEVAL` | nothing retrieved | broaden query / ingest docs |
| `WRONG_CHUNKS` | retrieved chunks don't ground the question | retrieval (reflexion targets this) |
| `UNSUPPORTED_GENERATION` | chunks ground it but the answer drifts | generation (prompt / temperature) |
| `ABSTAINED` | a correct refusal | none — safety valve working |
| `GROUNDED` | every claim supported | none |

## Measured result

```
mean faithfulness (served retrieve answers): 1.000  (target >= 0.75)
route accuracy: 1.000  | answered=6 direct=3 fallback=1 reflexion=2
```

## Honest scope

The eval set is a **day-scale curated set (10 queries)** to exercise every path,
not the 50-pair production set module 08 mandates for final scoring. The offline
faithfulness proxy is a lexical-support approximation of RAGAS faithfulness;
`USE_RAGAS=1` swaps in the real metric.
