# Day 10 — Concept Answers (Hallucination: causes + mitigation)

Prose answers to the five "understand" questions, each tied to the code and a
test that holds it true.

## 1. How to diagnose WHICH failure mode is active

"Hallucination" is not one bug — it is a family. Returning a single boolean
("hallucinated: yes/no") throws away the information needed to fix the pipeline.
The detector instead returns a **verdict** chosen by a fixed precedence, so the
most serious distrust signal is the one reported:

```
NO_RETRIEVAL > FABRICATED_CITATION > ABSTAINED > UNSUPPORTED > UNCITED > GROUNDED
```

The precedence matters. An off-topic answer that *also* invents a citation is
reported as `FABRICATED_CITATION`, not `UNSUPPORTED`, because the invented
citation is the louder alarm and points at a different remedy (the model is
manufacturing provenance, not merely drifting). Each verdict maps to a distinct
cause and therefore a distinct fix:

| Verdict | Root cause | Where to fix |
|---|---|---|
| `NO_RETRIEVAL` | Retriever found nothing above threshold | retrieval / index / chunking |
| `FABRICATED_CITATION` | Model invented a source label | generation prompt / decoding |
| `UNSUPPORTED` | Answer not entailed by chunks | generation grounding / context |
| `UNCITED` | Grounded but un-attributed | citation prompt |
| `ABSTAINED` | Correct refusal | nothing — working as intended |

*Code:* `src/scoring.py::assess`. *Test:*
`test_absent_context.py::test_five_scenarios_cover_four_distinct_failure_modes`.

## 2. Why a low retrieval score ≠ a bad answer (always)

The retrieval score is `query ↔ chunk` cosine similarity. It measures whether a
chunk *looks* relevant to the query — not whether the answer is faithful to it.
A chunk can score low (odd phrasing, short query, embedding quirk) yet still
contain the exact fact the answer extracts. Penalising the answer for that would
discard correct, grounded responses. So retrieval adequacy is weighted lowest
(0.15), and groundedness is computed against chunk **text**, not its score.

*Test:* `test_concepts.py::test_low_retrieval_score_does_not_imply_bad_answer`
(score 0.04 chunk, faithfully extracted → `GROUNDED`, served).

## 3. Why a high retrieval score ≠ a good answer (always)

Symmetric trap. A chunk can be a near-perfect query match while the *generated
answer* ignores it and asserts something from the model's parametric memory. The
retrieval step succeeded; the generation step hallucinated. Trusting the
retrieval score here is exactly how confident-but-wrong answers ship. Check 1
catches it by comparing the answer to the chunk content, independent of the
score.

*Test:* `test_concepts.py::test_high_retrieval_score_does_not_imply_good_answer`
(score 0.98 chunk, unrelated answer → `UNSUPPORTED`, fallback).

## 4. The 3-check pattern: retrieved? cited? faithful?

Three cheap, independent checks compose into one trustworthy decision:

1. **retrieved?** — Did the answer actually *use* the chunks? Each answer
   sentence (claim) is scored against every chunk by
   `max(token-containment, difflib contiguous-span)`. Containment (a set ratio
   over the claim's content tokens) is paraphrase-tolerant and length-fair;
   difflib's contiguous run rewards verbatim extraction. The headline is the
   fraction of claims that clear `τ_claim = 0.5`.
2. **cited?** — Are the `[Doc N]` citations *real*? Cited labels are compared to
   the actual chunk-id space; a label outside it (`[Doc 9]` when three chunks
   exist) is a fabricated citation — a hard red flag.
3. **faithful?** — Blend the signals and **refuse below 0.5**:
   `score = 0.60·support + 0.25·citation_validity + 0.15·retrieval_adequacy`.

*Code:* `src/checks.py`, `src/scoring.py`. *Tests:* `test_checks.py`.

## 5. What a "groundedness score" actually measures

It measures **answer ↔ evidence entailment**: can every claim in the answer be
traced back to the retrieved text? Three things it explicitly does *not* measure:

- **Not correctness.** "The capital of France is Paris" is true, but if it is not
  in the retrieved context the answer is `UNSUPPORTED` — the system cannot verify
  it from the evidence it was given, and a guardrail must not pretend otherwise.
  (*Test:* `test_groundedness_measures_evidence_not_world_correctness`.)
- **Not retrieval relevance.** That is `query ↔ chunk`; groundedness is
  `answer ↔ chunk`.
- **Not fluency.** A perfectly written paragraph with zero support scores ~0.

Groundedness is the one axis that isolates *hallucination* from every other
quality dimension — which is exactly why it is the metric a refusal gate should
be built on.
