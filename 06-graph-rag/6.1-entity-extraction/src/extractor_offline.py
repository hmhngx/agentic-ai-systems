"""Deterministic, offline entity + relation extraction with spaCy.

Pipeline:
  1. Run en_core_web_sm over each page's text.
  2. Entities = spaCy NER spans (mapped to our typed labels) UNION a domain
     lexicon for MODEL/DATASET/CONCEPT terms NER misses.
  3. Relations = for each sentence, every unordered pair of distinct entities that
     co-occur becomes a directed Triple in document order, labeled by the sentence's
     root-verb lemma (fallback "mentioned_with"), with the sentence as evidence.

No randomness anywhere -> identical output across runs (asserted in tests).
"""
from __future__ import annotations

import functools
import re

from src.loader import Page
from src.schema import Entity, EntityType, ExtractionResult, Triple

# Type specificity: domain types are more informative than generic NER types, so
# when the same surface form is seen as both (spaCy tags "NaturalQuestions" as ORG,
# the lexicon tags it DATASET) the domain type wins. Ties keep the first-seen type.
_SPECIFICITY = {
    EntityType.MODEL: 2,
    EntityType.DATASET: 2,
    EntityType.CONCEPT: 2,
    EntityType.WORK: 2,
    EntityType.PERSON: 1,
    EntityType.ORG: 1,
    EntityType.LOCATION: 1,
}

# spaCy NER label -> our typed label. Labels not listed are ignored.
_SPACY_MAP = {
    "PERSON": EntityType.PERSON,
    "ORG": EntityType.ORG,
    "GPE": EntityType.LOCATION,
    "LOC": EntityType.LOCATION,
    "FAC": EntityType.LOCATION,
    "PRODUCT": EntityType.MODEL,
    "WORK_OF_ART": EntityType.WORK,
}

# Junk surface forms that spaCy NER routinely emits from citations/abbreviations
# ("et al." -> "al" tagged GPE). Matched case-insensitively after cleaning.
_DENYLIST = {
    "al", "et al", "e.g", "i.e", "eg", "ie", "etc", "cf",
    "fig", "table", "section", "appendix", "eq", "vs", "inter alia",
    # prompt-template words this paper repeats ("Question:", "Answer:") that spaCy
    # mis-tags as ORG; they are not real entities here.
    "qa", "answer", "question", "answers", "questions",
}

# Domain lexicon: phrases spaCy won't tag, matched case-insensitively as whole words.
# Order longest-first so "GPT-3.5-Turbo" wins over a hypothetical "GPT".
_LEXICON: dict[str, EntityType] = {
    "GPT-3.5-Turbo": EntityType.MODEL,
    "GPT-3.5": EntityType.MODEL,
    "GPT-4": EntityType.MODEL,
    "Claude": EntityType.MODEL,
    "MPT-30B": EntityType.MODEL,
    "MPT-30B-Instruct": EntityType.MODEL,
    "LongChat": EntityType.MODEL,
    "LongChat-13B": EntityType.MODEL,
    "OpenAI": EntityType.ORG,
    "NaturalQuestions": EntityType.DATASET,
    "Natural Questions": EntityType.DATASET,
    "multi-document question answering": EntityType.CONCEPT,
    "key-value retrieval": EntityType.CONCEPT,
    "positional bias": EntityType.CONCEPT,
    "in-context learning": EntityType.CONCEPT,
}


@functools.lru_cache(maxsize=1)
def _nlp():
    import spacy

    # parser needed for sentence boundaries + root verb; ner for entities.
    return spacy.load("en_core_web_sm")


def _clean(name: str) -> str:
    # collapse whitespace/newlines that PDF extraction injects mid-name
    return re.sub(r"\s+", " ", name).strip(" .,;:")


def _lexicon_hits(text: str) -> list[tuple[int, int, str, EntityType]]:
    """Return (start, end, canonical_name, type) for each lexicon phrase in text.

    The surface form returned is the CANONICAL lexicon key (proper casing), not the
    matched text, so "mpt-30b" and "MPT-30B" collapse to one node. Boundaries exclude
    hyphens so "MPT-30B" does not match inside "MPT-30B-Instruct".
    """
    hits: list[tuple[int, int, str, EntityType]] = []
    for phrase in sorted(_LEXICON, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", text, flags=re.IGNORECASE):
            hits.append((m.start(), m.end(), phrase, _LEXICON[phrase]))
    return hits


def extract_offline(pages: list[Page]) -> ExtractionResult:
    nlp = _nlp()
    # name -> Entity (accumulates mentions); first-seen order preserved by dict
    entities: dict[str, Entity] = {}
    triples: list[Triple] = []
    seen_triples: set[tuple[str, str, str]] = set()

    def _register(name: str, etype: EntityType, *, curated: bool = False) -> str | None:
        name = _clean(name)
        if len(name) < 2 or not any(c.isalpha() for c in name):
            return None
        if not curated:
            # filter NER noise from citations and PDF front-matter artifacts.
            low = name.lower()
            if low in _DENYLIST or "et al" in low:
                return None
            if any(c.isdigit() for c in name):       # "1Stanford University", "Lin2"
                return None
            if len(name) <= 3 and name.islower() and " " not in name:
                return None
        if name in entities:
            existing = entities[name]
            existing.mentions += 1
            # curated lexicon always wins; otherwise only upgrade to a more specific type
            if curated or _SPECIFICITY[etype] > _SPECIFICITY[existing.type]:
                existing.type = etype
        else:
            entities[name] = Entity(name=name, type=etype)
        return name

    for page in pages:
        doc = nlp(page.text)
        for sent in doc.sents:
            sent_names: list[str] = []

            # spaCy NER spans in this sentence
            for ent in sent.ents:
                etype = _SPACY_MAP.get(ent.label_)
                if etype is None:
                    continue
                n = _register(ent.text, etype)
                if n:
                    sent_names.append(n)

            # domain lexicon spans in this sentence (curated -> bypass NER noise filter)
            for _s, _e, surface, etype in _lexicon_hits(sent.text):
                n = _register(surface, etype, curated=True)
                if n:
                    sent_names.append(n)

            # relation label = root verb lemma of the sentence
            root_verbs = [
                t.lemma_.lower()
                for t in sent
                if t.dep_ == "ROOT" and t.pos_ in {"VERB", "AUX"}
            ]
            relation = root_verbs[0] if root_verbs else "mentioned_with"

            # connect every unordered pair of distinct entities in the sentence
            uniq = list(dict.fromkeys(sent_names))  # de-dup, keep order
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = uniq[i], uniq[j]
                    key = (a, relation, b)
                    if key in seen_triples:
                        continue
                    seen_triples.add(key)
                    triples.append(
                        Triple(source=a, relation=relation, target=b, evidence=_clean(sent.text))
                    )

    return ExtractionResult(entities=list(entities.values()), triples=triples)
