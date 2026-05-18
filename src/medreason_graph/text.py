from __future__ import annotations

import math
import re
from collections.abc import Iterable

from medreason_graph.lexicon import ABBREVIATION_CONTEXT, AMBIGUOUS_ABBREVIATIONS, CONCEPTS, SECTION_KEYWORDS

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(normalize(text)) if token not in STOPWORDS]


def sentences(text: str) -> list[str]:
    parts = SENTENCE_RE.split(re.sub(r"\s+", " ", text.strip()))
    return [part.strip() for part in parts if part.strip()]


def sentence_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start = 0
    for match in SENTENCE_RE.finditer(text):
        _append_trimmed_span(spans, text, start, match.start())
        start = match.end()
    _append_trimmed_span(spans, text, start, len(text))
    return spans


def term_frequency(tokens: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def cosine_from_counts(left: dict[str, int], right: dict[str, int]) -> float:
    overlap = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in overlap)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def phrase_in_text(phrase: str, text: str) -> bool:
    normalized_text = normalize(text)
    normalized_phrase = normalize(phrase)
    if not normalized_phrase:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def detect_concepts(text: str, kind: str | None = None) -> set[str]:
    found: set[str] = set()
    for canonical, concept in CONCEPTS.items():
        if kind and concept.kind != kind:
            continue
        if any(_synonym_matches_concept(synonym, canonical, text) for synonym in concept.synonyms):
            found.add(canonical)
    return found


def canonicalize_term(text: str, kind: str | None = None, context: str = "") -> str | None:
    normalized = normalize(text)
    if normalized in AMBIGUOUS_ABBREVIATIONS:
        return _resolve_ambiguous_abbreviation(normalized, context)
    for canonical, concept in CONCEPTS.items():
        if kind and concept.kind != kind:
            continue
        if normalized == canonical or any(normalized == synonym for synonym in concept.synonyms):
            return canonical
    for canonical, concept in CONCEPTS.items():
        if kind and concept.kind != kind:
            continue
        if normalized in concept.synonyms or any(_synonym_matches_concept(synonym, canonical, normalized) for synonym in concept.synonyms):
            return canonical
    return None


def expand_query_terms(text: str) -> list[str]:
    expanded = tokenize(text)
    for canonical in detect_concepts(text):
        expanded.extend(tokenize(canonical))
        expanded.extend(tokenize(" ".join(CONCEPTS[canonical].synonyms)))
    return expanded


def detect_section_type(section_path: list[str]) -> str:
    joined = normalize(" ".join(section_path))
    for section_type, markers in SECTION_KEYWORDS.items():
        if any(marker in joined for marker in markers):
            return section_type
    return "unknown"


def _synonym_matches_concept(synonym: str, canonical: str, text: str) -> bool:
    if not phrase_in_text(synonym, text):
        return False
    normalized_synonym = normalize(synonym)
    if normalized_synonym not in AMBIGUOUS_ABBREVIATIONS:
        return True
    return _resolve_ambiguous_abbreviation(normalized_synonym, text) == canonical


def _resolve_ambiguous_abbreviation(abbreviation: str, context: str) -> str | None:
    candidates = AMBIGUOUS_ABBREVIATIONS.get(abbreviation, ())
    context_rules = ABBREVIATION_CONTEXT.get(abbreviation, {})
    for candidate in candidates:
        cues = context_rules.get(candidate, ())
        if any(phrase_in_text(cue, context) for cue in cues):
            return candidate
    return None


def _append_trimmed_span(spans: list[tuple[str, int, int]], text: str, start: int, end: int) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((text[start:end], start, end))
