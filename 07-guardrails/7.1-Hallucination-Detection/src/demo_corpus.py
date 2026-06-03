"""
demo_corpus.py - Deterministic, offline fixtures proving the Day-10 goal.

A tiny knowledge base about the Apollo 11 mission, plus five questions whose
answers are ABSENT from it. Each scenario hands the detector a pre-baked answer
that a careless RAG might emit, chosen so the five together exercise five
*different* failure shapes - not one trivial path:

    1. NO_RETRIEVAL         - nothing was retrieved at all.
    2. UNSUPPORTED          - off-topic chunks, confident uncited hallucination.
    3. FABRICATED_CITATION  - hallucination citing a [Doc 9] that does not exist.
    4. UNSUPPORTED          - cites a REAL [Doc 1], but the claim is not in it
                              (teaches: a valid citation does not make a claim
                              grounded).
    5. ABSTAINED            - the RAG correctly emitted its refusal phrase.

Guarantee: none of the five serve a confident hallucination; all five fall back.

A sixth CONTROL scenario (answer present and faithfully extracted) is included
so tests can prove the detector still SERVES a genuinely grounded answer - i.e.
the "all five fall back" result is discrimination, not a detector that always
refuses.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.types import RetrievedChunk, Verdict


# The knowledge base: three short, factual chunks about Apollo 11.
APOLLO_CHUNKS: list[RetrievedChunk] = [
    RetrievedChunk(
        chunk_id="Doc 1",
        text=(
            "Apollo 11 launched on July 16, 1969 from Launch Complex 39A at the "
            "Kennedy Space Center in Florida, atop a Saturn V rocket."
        ),
        score=0.31,
    ),
    RetrievedChunk(
        chunk_id="Doc 2",
        text=(
            "Neil Armstrong became the first human to walk on the Moon on July "
            "20, 1969, followed by Buzz Aldrin about twenty minutes later."
        ),
        score=0.29,
    ),
    RetrievedChunk(
        chunk_id="Doc 3",
        text=(
            "The Apollo 11 command module was named Columbia and the lunar "
            "module was named Eagle. Michael Collins remained in lunar orbit."
        ),
        score=0.27,
    ),
]


@dataclass(frozen=True)
class Scenario:
    """One labelled detector input, with the failure mode it should diagnose."""

    name: str
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    expected_verdict: Verdict
    expected_served: bool
    note: str


# The five absent-context scenarios. In every case the corpus contains no
# information that answers the question.
ABSENT_CONTEXT_SCENARIOS: list[Scenario] = [
    Scenario(
        name="empty_retrieval",
        question="What is the boiling point of water at sea level?",
        # A buggy RAG that generated despite retrieving nothing.
        answer="Water boils at 100 degrees Celsius at sea level.",
        chunks=[],
        expected_verdict=Verdict.NO_RETRIEVAL,
        expected_served=False,
        note="Retriever returned nothing; the question topic is absent entirely.",
    ),
    Scenario(
        name="offtopic_uncited_hallucination",
        question="How does photosynthesis convert sunlight into energy?",
        answer=(
            "Photosynthesis converts sunlight into chemical energy stored in "
            "glucose, using chlorophyll inside the chloroplasts of plant cells."
        ),
        chunks=APOLLO_CHUNKS,
        expected_verdict=Verdict.UNSUPPORTED,
        expected_served=False,
        note="Nearest chunks are off-topic; the answer shares no facts with them.",
    ),
    Scenario(
        name="fabricated_citation",
        question="Who wrote the novel Pride and Prejudice?",
        answer="Pride and Prejudice was written by Jane Austen in 1813 [Doc 9].",
        chunks=APOLLO_CHUNKS,
        expected_verdict=Verdict.FABRICATED_CITATION,
        expected_served=False,
        note="Cites [Doc 9] when only three chunks exist - an invented source.",
    ),
    Scenario(
        name="valid_citation_but_unfaithful",
        question="What is the capital of Australia?",
        answer="The capital of Australia is Canberra [Doc 1].",
        chunks=APOLLO_CHUNKS,
        expected_verdict=Verdict.UNSUPPORTED,
        expected_served=False,
        note="Cites a REAL chunk, but [Doc 1] is about the launch, not Australia.",
    ),
    Scenario(
        name="correct_abstention",
        question="What programming language ran the Apollo guidance computer?",
        answer=(
            "I do not have enough information in the retrieved documents to "
            "answer this question."
        ),
        chunks=APOLLO_CHUNKS,
        expected_verdict=Verdict.ABSTAINED,
        expected_served=False,
        note="The RAG correctly declined - a right action, not a hallucination.",
    ),
]


# A grounded control: the answer IS in the corpus and is faithfully extracted
# with a valid citation. Proves the detector serves genuine answers.
GROUNDED_CONTROL: Scenario = Scenario(
    name="grounded_control",
    question="When did Neil Armstrong first walk on the Moon?",
    answer="Neil Armstrong first walked on the Moon on July 20, 1969 [Doc 2].",
    chunks=APOLLO_CHUNKS,
    expected_verdict=Verdict.GROUNDED,
    expected_served=True,
    note="Answer is present in [Doc 2] and faithfully extracted - should serve.",
)
