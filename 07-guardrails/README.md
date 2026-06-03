# 07 — Guardrails

## Objective

This module covers safety and quality enforcement layers for LLM applications—input classification, output validation, PII detection, topic restriction, and framework comparison between NVIDIA NeMo Guardrails, Guardrails AI, and custom Pydantic validators. After completing this module, you will be able to build defense-in-depth guardrail pipelines that block malicious inputs, validate structured outputs, redact sensitive data, and enforce domain boundaries before responses reach end users.

## Core concepts

| Concept | Mental model (one sentence) | Why it matters in production |
|---------|----------------------------|------------------------------|
| Input classifiers | Pre-LLM filters that detect jailbreaks, prompt injection, and off-topic requests | Cheapest place to block attacks—one classification call vs. an entire generation pipeline |
| Output validators | Post-LLM checks ensuring responses match schema, stay on-topic, and contain no hallucinated citations | Catches model failures after generation; enables automatic retry with correction prompts |
| PII detection | Pattern matching and NER-based identification of names, emails, SSNs, and credit cards in inputs and outputs | Regulatory requirement (GDPR, HIPAA); a single PII leak in logs is a compliance incident |
| Topic restriction | Embedding-distance or classifier-based enforcement that queries stay within allowed domains | Prevents a customer support bot from answering medical or legal questions outside its scope |
| Framework tradeoffs | NeMo Guardrails (Colang DSL), Guardrails AI (RAIL spec), custom Pydantic (full control) | Each framework trades expressiveness for maintainability; choice depends on team skill and iteration speed |

## Implementation artifacts

| File | Description | Status |
|------|-------------|--------|
| [`7.1-Hallucination-Detection/`](7.1-Hallucination-Detection/README.md) | Day 10 — runtime groundedness guardrail wrapping any RAG: 3-check pattern (used-chunks? / citations-real? / faithful?), groundedness score 0–1, failure-mode verdict, and a fallback below threshold. 68 passing tests; offline-deterministic by default. Realizes the citation-verification + hallucination-flagging half of `output_validator.py`. | ✅ Done |
| `input_classifier.py` | Jailbreak and prompt-injection detection with configurable block/allow thresholds | 🔲 Pending |
| `output_validator.py` | Pydantic schema validation, citation verification, and hallucination flagging on LLM outputs | 🔲 Pending |
| `pii_detector.py` | Regex + Presidio-based PII detection with redaction before logging or storage | 🔲 Pending |
| `topic_guard.py` | Embedding-distance topic classifier restricting queries to allowed domains | 🔲 Pending |
| `guardrails_config.yaml` | Guardrails AI RAIL specification for combined input/output policies | 🔲 Pending |

## Key decisions & tradeoffs

- Custom Pydantic validators will be the default for output schema enforcement; Guardrails AI RAIL specs will be evaluated for teams preferring declarative policy files.
- Presidio will be chosen over regex-only PII detection for name and address recognition, with regex retained for structured identifiers (SSN, credit card, email).
- Input classification will run on every request before LLM invocation; output validation will run after generation with automatic single-retry on validation failure.
- Topic restriction will use embedding distance to domain centroid vectors rather than keyword blocklists—capturing paraphrased off-topic attempts.
- NeMo Guardrails will be documented as the choice for teams needing conversational flow control (dialog rails) beyond simple input/output filtering.

## Evaluation

| Metric | Target | Actual |
|--------|--------|--------|
| PII leak rate (in logs and responses) | 0% | |
| Off-topic block rate (true positives) | ≥ 99% | |
| False positive rate (legitimate queries blocked) | ≤ 2% | |
| Jailbreak block rate (known attack suite) | ≥ 95% | |

## References

- Guardrails AI documentation: https://www.guardrailsai.com/docs
- NVIDIA NeMo Guardrails documentation: https://docs.nvidia.com/nemo/guardrails/
- Microsoft Presidio: https://github.com/microsoft/presidio
- Pydantic validation documentation: https://docs.pydantic.dev/latest/concepts/validators/
- OWASP Top 10 for LLM Applications: https://owasp.org/www-project-top-10-for-large-language-model-applications/
