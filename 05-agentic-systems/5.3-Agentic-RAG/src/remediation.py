"""Map each RAGAS faithfulness failure mode to the concrete fix — the 'read a
RAGAS score table and know what to fix' skill, encoded."""
from __future__ import annotations

REMEDIATION: dict[str, str] = {
    "GROUNDED": "No action — every claim is supported by the retrieved context.",
    "ABSTAINED": "No action — a correct refusal is the safety valve working, not a bug.",
    "NO_RETRIEVAL": ("Fix RETRIEVAL: the query produced no in-vocabulary signal. Expand/"
                     "rewrite the query (reflexion), or ingest the missing documents."),
    "WRONG_CHUNKS": ("Fix RETRIEVAL: the retrieved chunks do not ground the question "
                     "(vocabulary mismatch / poor recall). Improve the query, embeddings, "
                     "or index — this is what the reflexion loop targets."),
    "UNSUPPORTED_GENERATION": ("Fix GENERATION: the chunks contain the answer but the model "
                               "added claims not in the context. Tighten the context-only "
                               "prompt and keep temperature at 0."),
    "UNCITED": "Fix GENERATION: require inline [Doc N] citations for every claim.",
}


def advise(failure_mode: str) -> str:
    return REMEDIATION.get(failure_mode, "Unknown failure mode — inspect the eval report.")
