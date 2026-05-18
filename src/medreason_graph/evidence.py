from __future__ import annotations

import hashlib
import logging

from medreason_graph.lexicon import CONCEPTS
from medreason_graph.logging_utils import log_event
from medreason_graph.models import EVIDENCE_CLAIM_SCHEMA, EvidenceClaim, PatientCase, RetrievalHit
from medreason_graph.text import canonicalize_term, detect_concepts, normalize, phrase_in_text, sentence_spans

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
RULE_OUT_CUES = ("rules out", "rule out", "excluded", "excludes", "exclude")
NON_RULE_OUT_CUES = (
    "does not rule out",
    "do not rule out",
    "doesn't rule out",
    "cannot rule out",
    "can not rule out",
    "does not exclude",
    "do not exclude",
    "cannot exclude",
    "can not exclude",
    "should not exclude",
)
TEST_CUES = ("obtain", "measure", "test", "evaluate", "initial evaluation", "serial", "perform")
STRONG_CUES = ("classic", "high risk", "emergent", "urgent", "must", "immediate")
WEAK_CUES = ("may", "can", "sometimes", "possible")
DIFFERENTIAL_ONLY_CUES = (
    "differential diagnosis",
    "differential includes",
    "differential should include",
    "differentiate between",
    "differentiate from",
    "differentiated from",
    "distinguish from",
    "other potential causes",
    "other conditions such as",
)
RED_FLAG_CUES = ("red flag", "emergent", "emergency", "urgent", "dangerous", "high risk")
EXTRACTION_METHOD = "deterministic_cue_v1"
logger = logging.getLogger(__name__)


def extract_evidence_claims(hits: list[RetrievalHit], case: PatientCase) -> list[EvidenceClaim]:
    patient_concepts = _case_concepts_by_status(case)
    claims: list[EvidenceClaim] = []
    seen: set[tuple[str, str, str | None, str, str]] = set()

    for hit in hits:
        for sentence, span_start, span_end in sentence_spans(hit.chunk.text):
            sentence_conditions = detect_concepts(sentence, kind="condition")
            sentence_findings = detect_concepts(sentence) - sentence_conditions
            if not sentence_conditions:
                continue
            polarity = _detect_polarity(sentence)
            strength = _detect_strength(sentence, hit.chunk.source_type)
            differential_only = _is_differential_only(sentence)

            for condition in sorted(sentence_conditions):
                present_matches = sorted(patient_concepts["present"] & sentence_findings)
                if not differential_only:
                    for finding in present_matches:
                        claim_type = _claim_type_for(sentence, polarity)
                        key = (condition, finding, polarity, hit.chunk.id, sentence)
                        _append_valid_claim(
                            claims=claims,
                            seen=seen,
                            key=key,
                            claim_type=claim_type,
                            condition=condition,
                            finding=finding,
                            polarity=polarity,
                            strength=strength,
                            hit=hit,
                            sentence=sentence,
                            span_start=span_start,
                            span_end=span_end,
                            confidence=_confidence_for(claim_type, polarity, strength),
                        )

                recommended_tests = sorted(_test_mentions(sentence))
                if recommended_tests and _has_test_cue(sentence):
                    for test_name in recommended_tests:
                        key = (condition, test_name, "recommends", hit.chunk.id, sentence)
                        _append_valid_claim(
                            claims=claims,
                            seen=seen,
                            key=key,
                            claim_type="requires_test",
                            condition=condition,
                            finding=test_name,
                            polarity="recommends",
                            strength=strength,
                            hit=hit,
                            sentence=sentence,
                            span_start=span_start,
                            span_end=span_end,
                            confidence=_confidence_for("requires_test", "recommends", strength),
                        )
    log_event(logger, "evidence_extracted", hits=len(hits), claims=len(claims))
    return claims


def validate_evidence_claim(claim: EvidenceClaim, source_text: str | None = None) -> list[str]:
    errors: list[str] = []
    properties = EVIDENCE_CLAIM_SCHEMA["properties"]
    if claim.schema_version != properties["schema_version"]["const"]:
        errors.append("schema_version")
    if claim.claim_type not in properties["claim_type"]["enum"]:
        errors.append("claim_type")
    if claim.polarity not in properties["polarity"]["enum"]:
        errors.append("polarity")
    if claim.strength not in properties["strength"]["enum"]:
        errors.append("strength")
    if not claim.condition:
        errors.append("condition")
    if not claim.source_id:
        errors.append("source_id")
    if not claim.sentence.strip():
        errors.append("sentence")
    if claim.source_span_start < 0 or claim.source_span_end <= claim.source_span_start:
        errors.append("source_span")
    if not claim.source_text_hash:
        errors.append("source_text_hash")
    if not 0.0 <= claim.extraction_confidence <= 1.0:
        errors.append("extraction_confidence")
    if not claim.extraction_method:
        errors.append("extraction_method")
    if source_text is not None:
        if claim.source_span_end > len(source_text):
            errors.append("source_span_bounds")
        else:
            span_text = source_text[claim.source_span_start:claim.source_span_end]
            if span_text != claim.sentence:
                errors.append("source_span_text")
            if _source_text_hash(source_text) != claim.source_text_hash:
                errors.append("source_text_hash_mismatch")
    return errors


def _case_concepts_by_status(case: PatientCase) -> dict[str, set[str]]:
    concepts = {
        "present": detect_concepts(case.chief_complaint) | detect_concepts(case.free_text),
        "absent": set(),
        "missing": set(),
    }
    context = " ".join([case.chief_complaint, case.free_text, " ".join(finding.name for finding in case.findings)])
    for finding in case.findings:
        canonical = finding.concept or canonicalize_term(finding.name, context=context)
        if canonical and finding.status in concepts:
            concepts[finding.status].add(canonical)
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
    if any(cue in lowered for cue in NON_RULE_OUT_CUES):
        return "supports"
    if any(cue in lowered for cue in RULE_OUT_CUES) or any(cue in lowered for cue in AGAINST_CUES):
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


def _claim_type_for(text: str, polarity: str) -> str:
    lowered = normalize(text)
    if polarity == "argues_against" and any(cue in lowered for cue in RULE_OUT_CUES):
        return "rules_out"
    if polarity == "supports" and any(cue in lowered for cue in RED_FLAG_CUES):
        return "red_flag"
    return "supports" if polarity == "supports" else "argues_against"


def _confidence_for(claim_type: str, polarity: str, strength: str) -> float:
    base = {
        "strong": 0.86,
        "moderate": 0.72,
        "weak": 0.58,
    }.get(strength, 0.5)
    if claim_type == "requires_test":
        base += 0.04
    if claim_type == "argues_against" and polarity == "argues_against":
        base -= 0.05
    return max(0.0, min(round(base, 2), 1.0))


def _append_valid_claim(
    *,
    claims: list[EvidenceClaim],
    seen: set[tuple[str, str, str | None, str, str]],
    key: tuple[str, str, str | None, str, str],
    claim_type: str,
    condition: str,
    finding: str | None,
    polarity: str,
    strength: str,
    hit: RetrievalHit,
    sentence: str,
    span_start: int,
    span_end: int,
    confidence: float,
) -> None:
    if key in seen:
        return
    claim = _claim(
        claim_type=claim_type,
        condition=condition,
        finding=finding,
        polarity=polarity,
        strength=strength,
        hit=hit,
        sentence=sentence,
        span_start=span_start,
        span_end=span_end,
        confidence=confidence,
    )
    errors = validate_evidence_claim(claim, hit.chunk.text)
    if errors:
        log_event(logger, "evidence_claim_rejected", claim_id=claim.id, errors=errors)
        return
    seen.add(key)
    claims.append(claim)


def _claim(
    *,
    claim_type: str,
    condition: str,
    finding: str | None,
    polarity: str,
    strength: str,
    hit: RetrievalHit,
    sentence: str,
    span_start: int,
    span_end: int,
    confidence: float,
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
        source_span_start=span_start,
        source_span_end=span_end,
        source_text_hash=_source_text_hash(hit.chunk.text),
        extraction_confidence=confidence,
        extraction_method=EXTRACTION_METHOD,
    )


def _claim_id(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"ev_{digest}"


def _source_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
