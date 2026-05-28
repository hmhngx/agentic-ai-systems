"""
generator.py - Prompt construction and Claude API call.

Design decision: raw anthropic.Anthropic() client, zero framework wrappers.
This forces explicit control over every parameter passed to the API.

Design decision: temperature=0.
Reason: factual RAG requires deterministic extraction, not creative generation.
Higher temperatures flatten the token probability distribution, allowing the
model to sample from lower-probability (often hallucinated) tokens.
At temperature=0, the model deterministically selects the highest-probability
token at each step - maximizing adherence to the retrieved context.
Reference: TempPerturb-Eval framework (see Obsidian note).

Design decision: max_tokens=1024 for answers.
Sufficient for detailed answers with inline citations.
Prevents runaway generation that could mix topics.

Design decision: model="claude-sonnet-4-20250514"
Reason: Claude Sonnet 4 balances quality and cost for this use case.
For production with strict latency requirements: claude-haiku-4-5-20251001.
For maximum quality: claude-opus-4-0.
"""

from __future__ import annotations

import anthropic

from src.citation import format_context_block, validate_citations


MODEL: str = "claude-sonnet-4-20250514"
MAX_TOKENS: int = 1024
TEMPERATURE: int = 0      # MANDATORY for factual RAG - see docstring above. int 0 (not 0.0) for clarity.


def build_system_prompt() -> str:
    """Return the RAG system prompt - the behavior contract for Claude.

    The string below must contain all six required behavior elements:
        1. Role definition
        2. Context-only constraint (no parametric knowledge supplementation)
        3. [Doc N] citation requirement
        4. No-information escape hatch with the exact phrase
        5. Conflict surfacing instruction
        6. Scope constraint (no inference/extrapolation)
    Removing any element degrades faithfulness in measurable ways - see
    the Day 4 Obsidian note for the eval evidence.
    """
    return (
        # 1. Role definition - frames the model as an extractor, not an author.
        "You are an expert research assistant whose only job is to answer "
        "questions using a provided set of CONTEXT documents retrieved from "
        "a knowledge base.\n\n"
        # 2. Hard constraint: context-only, no parametric supplementation.
        "HARD CONSTRAINTS:\n"
        "- Answer ONLY using the provided CONTEXT blocks labeled "
        "[Doc 1], [Doc 2], etc.\n"
        "- Do NOT use your pre-trained knowledge to supplement the answer.\n"
        "- Do NOT introduce facts not present in the CONTEXT.\n\n"
        # 3. Citation requirement - inline [Doc N] for every claim.
        "CITATIONS:\n"
        "- Cite every claim or sentence with the document label, e.g. "
        "[Doc 1].\n"
        "- If multiple documents support a claim, cite all of them: "
        "[Doc 1][Doc 3].\n"
        "- Citations must be inline, immediately after the claim they "
        "support.\n\n"
        # 4. No-information escape hatch - the exact response phrase is mandatory.
        "WHEN INFORMATION IS MISSING:\n"
        "If the CONTEXT does not contain enough information to answer the "
        "question, respond with exactly:\n"
        "'I do not have enough information in the retrieved documents to "
        "answer this question.'\n"
        "followed by a brief explanation of what information is missing. "
        "Do not attempt to fill the gap from outside knowledge.\n\n"
        # 5. Conflict surfacing - never silently average contradictions.
        "CONFLICTS:\n"
        "If documents contradict each other, surface the contradiction "
        "explicitly: '[Doc 1] states X, but [Doc 2] states Y.' Do not "
        "resolve contradictions by averaging or choosing one silently.\n\n"
        # 6. Scope constraint - no inference beyond explicit statements.
        "SCOPE:\n"
        "Do not infer, extrapolate, or generalize beyond what is explicitly "
        "stated in the CONTEXT. Stick to what the documents say."
    )


def build_user_message(query: str, context_block: str) -> str:
    """Return the user-turn message: CONTEXT first, then QUERY.

    Why CONTEXT before QUERY?
        The "lost in the middle" phenomenon - LLMs attend most strongly
        to tokens at the END of the context window (recency effect).
        Placing the QUERY at the very end means it receives maximum
        attention. The most relevant chunk is already at position [Doc 1]
        in the context block (highest-score first), so the best evidence
        ends up adjacent to the query.

        Reversing this (QUERY before CONTEXT) is a known anti-pattern
        that reduces factual grounding. Do not change the order.
    """
    return f"CONTEXT:\n{context_block}\n\nQUERY:\n{query}"


def generate_answer(query: str, results: list[dict]) -> tuple[str, list[str]]:
    """Call Claude with the assembled prompt and return ``(answer, warnings)``.

    Steps:
        1. Format the context block from retrieved results.
        2. Build the system prompt (behavior contract).
        3. Build the user message (CONTEXT before QUERY).
        4. Call ``anthropic.Anthropic().messages.create`` with deterministic
           parameters (temperature=0).
        5. Extract the answer text from ``response.content[0].text``.
        6. Run citation validation against the original results list.

    On ``anthropic.APIError`` we return a graceful error message instead
    of crashing - the CLI loop should survive a transient API hiccup so
    the user can retry without restarting ingestion.
    """
    context_block: str = format_context_block(results)
    system_prompt: str = build_system_prompt()
    user_message: str = build_user_message(query, context_block)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    try:
        response = client.messages.create(
            model=MODEL,                  # claude-sonnet-4-20250514: quality/cost balance for RAG
            max_tokens=MAX_TOKENS,        # 1024: enough for detailed cited answers, caps runaway generation
            temperature=TEMPERATURE,      # 0: deterministic top-token selection - mandatory for factual RAG
            system=system_prompt,         # behavior contract (6 mandatory elements)
            messages=[
                {"role": "user", "content": user_message},  # CONTEXT then QUERY, single turn
            ],
        )
    except anthropic.APIError as exc:
        # Specific exception class - never bare ``except``. We return the
        # error in the answer slot so the CLI displays it cleanly; an
        # empty warnings list keeps the citation-warning channel quiet.
        error_message: str = f"I encountered an error calling the Claude API: {exc}"
        return error_message, []

    # response.content is a list of content blocks; for a plain text
    # answer with no tool use it has exactly one TextBlock at index 0.
    answer_text: str = response.content[0].text

    warnings: list[str] = validate_citations(answer_text, results)
    return answer_text, warnings
