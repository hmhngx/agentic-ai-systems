# 7.3 — Output Guard

A **post-generation output validation pipeline** that intercepts every LLM answer
before it reaches the user. Three independent stages run in series; the first
`BLOCK` short-circuits the chain. A `RETRY` verdict from Stage 3 triggers one
regeneration attempt with an enriched prompt before final blocking.

```
answer + context
       |
       v
+--------------------------------------------+
|  Stage 1: FormatChecker    [<5ms, structural]  |
|  empty? truncated? Pydantic min_length fail?   |
|  -> BLOCK immediately                          |
+------------------+-------------------------+
                   | pass
                   v
+--------------------------------------------+
|  Stage 2: ToxicityChecker  [<5ms, keyword]     |
|  30 patterns: violence, hate, explicit,        |
|  self_harm. Hard match -> 0.95 BLOCK.          |
|  Soft count x 0.30 -> BLOCK if > 0.50.        |
+------------------+-------------------------+
                   | pass
                   v
+--------------------------------------------+
|  Stage 3: FaithfulnessChecker  [<20ms offline] |
|  Token containment: |ans^ctx| / |ans|          |
|  score >= 0.60 -> ALLOW                        |
|  score <  0.60 -> RETRY (orchestrator retries) |
+------------------+-------------------------+
                   |
                   v
           OutputGuardReport
```

`GuardedRAG` in `guardrail_layer.py` wires `InputGuard` (7.2) +
generation + `OutputGuard` into a single callable, retrying once
when faithfulness is borderline and logging every block to
`logs/guardrail_blocks.log`.

---

## Results

```
[unfaithful-1]  verdict : BLOCKED [retry fired]
[unfaithful-2]  verdict : BLOCKED [retry fired]
[unfaithful-3]  verdict : BLOCKED [retry fired]
[unfaithful-4]  verdict : BLOCKED [retry fired]
[unfaithful-5]  verdict : BLOCKED [retry fired]
[borderline-retry] verdict : ALLOWED [retry fired]  faith: 0.429 -> 0.895
[faithful]      verdict : ALLOWED
[injection]     verdict : BLOCKED  (output guard - format)
```

| Metric | Result | Target |
|--------|--------|--------|
| Unfaithful answers blocked | **5/5** | 5/5 |
| Retry fires and improves score | **yes** (0.429->0.895) | yes |
| Full test suite | **64 passed** | all pass |

---

## Quickstart

```bash
cd 07-guardrails/7.3-Output-Guard
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Full offline demo
python guardrail_layer.py --demo

# JSON output
python guardrail_layer.py --demo --json

# Run the test suite
python -m pytest -v
```

### Programmatic use

```python
from guardrail_layer import GuardedRAG, GuardedRAGResult

guarded = GuardedRAG()   # auto-wires InputGuard + OutputGuard
result = guarded.query(
    "What is RAG?",
    context=["RAG retrieves documents from a vector database..."],
)

if result.blocked:
    print(f"Blocked: {result.reason}")
else:
    print(result.answer)
    if result.retry_fired:
        print("(answer was produced after one retry)")
```

---

## Architecture

| File | Responsibility |
|------|---------------|
| `guardrail_layer.py` | `GuardedRAG`, `GuardedRAGResult`, retry logic, logging, demo, CLI |
| `src/guard.py` | `OutputGuard.check()` — 3-stage waterfall, guardrails-ai schema validation |
| `src/format_checker.py` | Stage 1: empty/truncated/Pydantic checks |
| `src/toxicity_checker.py` | Stage 2: keyword patterns + optional LLM |
| `src/faithfulness_checker.py` | Stage 3: token containment + optional LLM |
| `src/llm_judge.py` | OpenRouter escalation (lazy import, raises `LLMJudgeError`) |
| `src/types.py` | `Verdict`, `CheckResult`, `OutputGuardReport`, `OutputGuardConfig` |

### guardrails-ai integration

`Guard.for_pydantic(OutputCheckResultModel)` wraps each `CheckResult` before it is
appended to the report. Any out-of-range score or invalid verdict raises
`GuardrailInternalError` immediately — the same schema-contract pattern as 7.2.

### Why token containment (not RAGAS)?

RAGAS requires 2+ extra LLM calls per query (decompose claims + verify each) —
500ms to 2s added latency. Token containment (`|answer_tokens ^ context_tokens| / |answer_tokens|`)
runs in <20ms, catches invented vocabulary, and uses the same algorithm as 7.1.
The LLM judge (`USE_LLM=1`) resolves genuinely borderline scores without paying
the overhead on every query.
