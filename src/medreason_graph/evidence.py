from __future__ import annotations

import hashlib
import logging

from medreason_graph.lexicon import CONCEPTS
from medreason_graph.logging_utils import log_event
from medreason_graph.models import EvidenceClaim, PatientCase, PatientFinding, RetrievalHit
from medreason_graph.text import canonicalize_term, detect_concepts, normalize, phrase_in_text, sentences

SUPPORT_CUES = (
    "suggest",
    "support",
    "consistent with",
    "associated with",
    "typical",
    "classic",
    "presents with",
    "presentation includes",
    "concerning for",
)
AGAINST_CUES = ("argues against", "argue against", "less likely", "less typical", "unlikely", "absence of", "without")
TEST_CUES = ("obtain", "measure", "test", "evaluate", "initial evaluation", "serial", "perform")
STRONG_CUES = ("classic", "high risk", "emergent", "urgent", "must", "immediate")
WEAK_CUES = ("may", "can", "sometimes", "possible")
DIFFERENTIAL_ONLY_CUES = (
    "differential diagnosis",
    "differential includes",
    "differential should include",
    "differentiate between",
    "differentiate from",
    "other potential causes",
    "other conditions such as",
)
logger = logging.getLogger(__name__)


def extract_evidence_claims(hits: list[RetrievalHit], case: PatientCase) -> list[EvidenceClaim]:
    patient_concepts = _case_concepts(case)
    claims: list[EvidenceClaim] = []
    seen: set[tuple[str, str, str | None, str, str]] = set()

    for hit in hits:
        for sentence in sentences(hit.chunk.text):
            sentence_conditions = detect_concepts(sentence, kind="condition")
            sentence_findings = detect_concepts(sentence) - sentence_conditions
            if not sentence_conditions:
                continue
            polarity = _detect_polarity(sentence)
            strength = _detect_strength(sentence, hit.chunk.source_type)
            differential_only = _is_differential_only(sentence)

            for condition in sorted(sentence_conditions):
                patient_matches = sorted(patient_concepts & sentence_findings)
                if patient_matches and not differential_only:
                    for finding in patient_matches:
                        key = (condition, finding, polarity, hit.chunk.id, sentence)
                        if key in seen:
                            continue
                        seen.add(key)
                        claims.append(
                            _claim(
                                claim_type="finding_supports_condition" if polarity == "supports" else "finding_argues_against_condition",
                                condition=condition,
                                finding=finding,
                                polarity=polarity,
                                strength=strength,
                                hit=hit,
                                sentence=sentence,
                            )
                        )

                recommended_tests = sorted(_test_mentions(sentence))
                if recommended_tests and _has_test_cue(sentence):
                    for test_name in recommended_tests:
                        key = (condition, test_name, "recommends", hit.chunk.id, sentence)
                        if key in seen:
                            continue
                        seen.add(key)
                        claims.append(
                            _claim(
                                claim_type="recommends_test",
                                condition=condition,
                                finding=test_name,
                                polarity="recommends",
                                strength=strength,
                                hit=hit,
                                sentence=sentence,
                            )
                        )
    log_event(logger, "evidence_extracted", hits=len(hits), claims=len(claims))
    return claims


def _case_concepts(case: PatientCase) -> set[str]:
    concepts = detect_concepts(case.chief_complaint)
    concepts.update(detect_concepts(case.free_text))
    context = " ".join([case.chief_complaint, case.free_text, " ".join(finding.name for finding in case.findings)])
    for finding in case.findings:
        canonical = finding.concept or canonicalize_term(finding.name, context=context)
        if canonical and finding.status == "present":
            concepts.add(canonical)
    return concepts


def _test_mentions(text: str) -> set[str]:
    return {
        canonical
        for canonical, concept in CONCEPTS.items()
        if concept.kind == "test" and any(phrase_in_text(synonym, text) for synonym in concept.synonyms)
    }


def _has_test_cue(text: str) -> bool:
    lowered = normalize(text)
    return any(cue in lowered for cue in TEST_CUES)


def _detect_polarity(text: str) -> str:
    lowered = normalize(text)
    if any(cue in lowered for cue in AGAINST_CUES):
        return "argues_against"
    return "supports"


def _detect_strength(text: str, source_type: str) -> str:
    lowered = normalize(text)
    if source_type in {"guideline", "drug_label"} or any(cue in lowered for cue in STRONG_CUES):
        return "strong"
    if any(cue in lowered for cue in WEAK_CUES):
        return "weak"
    return "moderate"


def _is_differential_only(text: str) -> bool:
    lowered = normalize(text)
    return any(cue in lowered for cue in DIFFERENTIAL_ONLY_CUES)


def _claim(
    *,
    claim_type: str,
    condition: str,
    finding: str | None,
    polarity: str,
    strength: str,
    hit: RetrievalHit,
    sentence: str,
) -> EvidenceClaim:
    claim_id = _claim_id(condition, finding or "", polarity, hit.chunk.id, sentence)
    return EvidenceClaim(
        id=claim_id,
        claim_type=claim_type,
        condition=condition,
        finding=finding,
        polarity=polarity,
        strength=strength,
        source_id=hit.chunk.id,
        source_type=hit.chunk.source_type,
        source_title=hit.chunk.title,
        section_path=hit.chunk.section_path,
        paragraph_index=hit.chunk.paragraph_index,
        sentence=sentence,
    )


def _claim_id(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"ev_{digest}"
