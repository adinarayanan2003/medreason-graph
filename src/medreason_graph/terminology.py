from __future__ import annotations

from dataclasses import dataclass

from medreason_graph.lexicon import ABBREVIATION_CONTEXT, AMBIGUOUS_ABBREVIATIONS, CONCEPTS
from medreason_graph.text import normalize, phrase_in_text


@dataclass(frozen=True)
class NormalizedConcept:
    text: str
    canonical: str | None
    kind: str | None
    semantic_type: str | None
    match_method: str
    ambiguous: bool = False
    candidates: tuple[str, ...] = ()


def normalize_medical_term(text: str, kind: str | None = None, context: str = "") -> NormalizedConcept:
    normalized = normalize(text)
    context_text = f"{text} {context}".strip()
    if normalized in AMBIGUOUS_ABBREVIATIONS:
        candidates = AMBIGUOUS_ABBREVIATIONS[normalized]
        resolved = _resolve_ambiguous_abbreviation(normalized, context_text, candidates)
        if not resolved:
            return NormalizedConcept(
                text=text,
                canonical=None,
                kind=None,
                semantic_type=None,
                match_method="ambiguous_abbreviation",
                ambiguous=True,
                candidates=candidates,
            )
        concept = CONCEPTS.get(resolved)
        return NormalizedConcept(
            text=text,
            canonical=resolved,
            kind=concept.kind if concept else None,
            semantic_type=concept.semantic_type if concept else None,
            match_method="contextual_abbreviation",
            ambiguous=False,
            candidates=candidates,
        )

    for canonical, concept in CONCEPTS.items():
        if kind and concept.kind != kind:
            continue
        if normalized == canonical:
            return NormalizedConcept(text, canonical, concept.kind, concept.semantic_type, "canonical")
        if any(normalized == normalize(synonym) for synonym in concept.synonyms):
            method = "abbreviation" if len(normalized) <= 4 else "synonym"
            return NormalizedConcept(text, canonical, concept.kind, concept.semantic_type, method)

    for canonical, concept in CONCEPTS.items():
        if kind and concept.kind != kind:
            continue
        if any(phrase_in_text(synonym, normalized) for synonym in concept.synonyms):
            return NormalizedConcept(text, canonical, concept.kind, concept.semantic_type, "phrase")

    return NormalizedConcept(text, None, None, None, "none")


def detect_normalized_concepts(text: str, kind: str | None = None) -> list[NormalizedConcept]:
    matches: list[NormalizedConcept] = []
    for canonical, concept in CONCEPTS.items():
        if kind and concept.kind != kind:
            continue
        for synonym in concept.synonyms:
            normalized_synonym = normalize(synonym)
            if not phrase_in_text(normalized_synonym, text):
                continue
            normalized = normalize_medical_term(synonym, kind=kind, context=text)
            if normalized.canonical == canonical:
                matches.append(normalized)
            break
    return matches


def _resolve_ambiguous_abbreviation(abbreviation: str, context: str, candidates: tuple[str, ...]) -> str | None:
    context_rules = ABBREVIATION_CONTEXT.get(abbreviation, {})
    for candidate in candidates:
        cues = context_rules.get(candidate, ())
        if any(phrase_in_text(cue, context) for cue in cues):
            return candidate
    return None

