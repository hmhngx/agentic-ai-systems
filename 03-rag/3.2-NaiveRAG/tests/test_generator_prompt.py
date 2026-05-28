"""Offline tests for ``src/generator.py``'s prompt construction.

We do NOT call Claude here. The Anthropic API is mocked out by
simply never invoking ``generate_answer``; these tests only touch
``build_system_prompt``, ``build_user_message``, and the module-level
constants.

The system prompt is the BEHAVIOR CONTRACT for the LLM. Each
missing element below is a documented hallucination vector:

  1. Role definition           -> without it, model writes essays, not answers
  2. Context-only constraint   -> without it, model uses parametric memory
  3. [Doc N] citation rule     -> without it, no audit trail
  4. No-info escape hatch      -> without it, model confabulates on miss
  5. Conflict surfacing        -> without it, model silently averages
  6. No-extrapolation rule     -> without it, model reasons beyond evidence

Lost-in-the-middle: query MUST come AFTER context. Reversing this
order is a known anti-pattern that reduces factual grounding.
"""

from __future__ import annotations


def test_build_system_prompt_contains_role() -> None:
    """System prompt must define a role - frames the model as extractor."""
    from src.generator import build_system_prompt
    prompt = build_system_prompt()
    assert isinstance(prompt, str), \
        f"build_system_prompt must return str, got {type(prompt).__name__}."
    assert len(prompt) > 200, \
        f"System prompt only {len(prompt)} chars - too short to contain all " \
        "6 mandatory behavior elements. A trimmed prompt loses elements " \
        "silently and the hallucination rate goes up without any warning."
    assert "assistant" in prompt.lower() or "expert" in prompt.lower(), \
        "Role definition missing from system prompt. " \
        "Without a role anchor, the LLM defaults to creative-writing mode " \
        "and treats CONTEXT as inspiration rather than ground truth."


def test_build_system_prompt_contains_context_only_constraint() -> None:
    """The single most important constraint: answer ONLY from CONTEXT."""
    from src.generator import build_system_prompt
    prompt_lower = build_system_prompt().lower()
    assert "only" in prompt_lower and "context" in prompt_lower, \
        "Context-only constraint missing - this is THE constraint that " \
        "prevents the LLM from supplementing answers with its pre-trained " \
        "knowledge. Removing it turns the RAG system into a fancy chatbot."


def test_build_system_prompt_contains_citation_requirement() -> None:
    """LLM must be told to inline [Doc N] - otherwise we cannot validate citations."""
    from src.generator import build_system_prompt
    prompt_lower = build_system_prompt().lower()
    assert "[doc" in prompt_lower, \
        "System prompt missing the literal '[Doc' token. " \
        "validate_citations greps for [Doc N] - without that token in the " \
        "prompt the LLM picks an arbitrary citation format and our parser misses it."
    assert "cite" in prompt_lower or "citation" in prompt_lower, \
        "System prompt missing 'cite'/'citation' instruction. " \
        "Showing labels without demanding their use produces uncited answers."


def test_build_system_prompt_contains_escape_hatch() -> None:
    """LLM must be given an explicit 'I do not have enough information' phrase."""
    from src.generator import build_system_prompt
    prompt_lower = build_system_prompt().lower()
    assert (
        "do not have enough information" in prompt_lower
        or "insufficient" in prompt_lower
        or "cannot answer" in prompt_lower
    ), "No-information escape hatch MISSING - without an explicit refusal " \
       "phrase the LLM tries to answer from parametric memory whenever " \
       "retrieval is weak. This is the #1 hallucination vector in RAG."


def test_build_system_prompt_contains_conflict_surfacing() -> None:
    """When chunks contradict, LLM must surface it - not silently average."""
    from src.generator import build_system_prompt
    prompt_lower = build_system_prompt().lower()
    assert "conflict" in prompt_lower or "contradict" in prompt_lower, \
        "Conflict-surfacing instruction missing. Without it the LLM picks " \
        "one side of a contradiction silently or averages them, hiding " \
        "data-quality issues from the user."


def test_build_system_prompt_contains_no_extrapolation() -> None:
    """LLM must be constrained from inferring beyond explicit context statements."""
    from src.generator import build_system_prompt
    prompt_lower = build_system_prompt().lower()
    assert (
        "infer" in prompt_lower
        or "extrapolate" in prompt_lower
        or "generalize" in prompt_lower
        or "beyond" in prompt_lower
    ), "No-extrapolation constraint missing. Without it the model strays " \
       "from 'what the doc says' to 'what the doc implies', which is the " \
       "boundary between RAG and hallucination."


def test_build_system_prompt_no_parametric_knowledge() -> None:
    """LLM must be explicitly told NOT to use pre-trained knowledge."""
    from src.generator import build_system_prompt
    prompt_lower = build_system_prompt().lower()
    assert (
        "pre-trained" in prompt_lower
        or "training" in prompt_lower
        or "knowledge" in prompt_lower
    ), "Parametric-memory suppression missing from system prompt. " \
       "Without an explicit 'don't use pre-trained knowledge' clause, the " \
       "LLM defaults to mixing context with what it 'knows', which is " \
       "indistinguishable from hallucination at the user's end."


def test_build_user_message_context_before_query() -> None:
    """CONTEXT must precede QUERY - reversing it triggers lost-in-the-middle."""
    from src.generator import build_user_message
    context = "This is context text about machine learning."
    query = "What is backpropagation?"
    message = build_user_message(query, context)
    assert isinstance(message, str), \
        f"build_user_message must return str, got {type(message).__name__}."

    context_pos = message.find("context text about machine learning")
    query_pos = message.find("What is backpropagation")
    assert context_pos != -1, \
        "Context text not found in user message - either dropped or " \
        "mangled before reaching the LLM."
    assert query_pos != -1, \
        "Query text not found in user message - LLM has nothing to answer."
    assert context_pos < query_pos, \
        f"CRITICAL: query (pos {query_pos}) appears before context " \
        f"(pos {context_pos}). This violates lost-in-the-middle " \
        "placement - the LLM attends most to the END of the prompt, " \
        "so the query must be last for maximum grounding attention."


def test_build_user_message_contains_both() -> None:
    """Both context and query must be present - missing either breaks the turn."""
    from src.generator import build_user_message
    context = "This is context text about machine learning."
    query = "What is backpropagation?"
    message = build_user_message(query, context)
    assert context[:20] in message, \
        "Context prefix missing from user message. " \
        "An empty CONTEXT forces the model to answer from parametric memory."
    assert query in message, \
        "Query missing from user message. " \
        "Without a query the LLM has no goal and produces a summary at best."


def test_temperature_is_zero() -> None:
    """temperature=0 is mandatory for factual RAG - non-zero flattens token probs."""
    from src.generator import TEMPERATURE
    assert TEMPERATURE == 0, \
        f"TEMPERATURE must be 0 for factual RAG, got {TEMPERATURE}. " \
        "Non-zero temperature flattens token probabilities and lets the " \
        "model sample lower-probability tokens, which is exactly the " \
        "hallucination vector we are trying to close."


def test_temperature_is_int_not_float() -> None:
    """TEMPERATURE must be the literal int 0 - documented choice for clarity."""
    from src.generator import TEMPERATURE
    assert isinstance(TEMPERATURE, int) and not isinstance(TEMPERATURE, bool), \
        f"TEMPERATURE should be int 0, not {type(TEMPERATURE).__name__} " \
        f"(value={TEMPERATURE!r}). The int form makes deterministic intent " \
        "obvious in code review; a stray 0.0 invites someone to 'just bump " \
        "it a little' in the future."


def test_max_tokens_correct() -> None:
    """max_tokens=1024 caps runaway generation while leaving room for citations."""
    from src.generator import MAX_TOKENS
    assert MAX_TOKENS == 1024, \
        f"MAX_TOKENS should be 1024, got {MAX_TOKENS}. " \
        "Too small truncates cited answers mid-sentence; too large lets " \
        "the model drift across topics once it runs out of grounded content."


def test_model_is_claude_sonnet() -> None:
    """Model identifier must be a Claude (Sonnet/Opus/Haiku) deployment."""
    from src.generator import MODEL
    assert "claude" in MODEL.lower(), \
        f"MODEL should be a Claude model, got {MODEL!r}. " \
        "The Anthropic SDK is hardcoded - any other vendor breaks the API call."
    assert (
        "sonnet" in MODEL.lower()
        or "opus" in MODEL.lower()
        or "haiku" in MODEL.lower()
    ), f"MODEL should pin a specific Claude tier (sonnet/opus/haiku), " \
       f"got {MODEL!r}. A vague model id risks silent upgrades that " \
       "change cost, latency, and hallucination behavior without notice."
