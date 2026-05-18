from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import subprocess
from typing import Any

from medreason_graph.lexicon import CONCEPTS
from medreason_graph.logging_utils import log_event
from medreason_graph.models import EVIDENCE_CLAIM_SCHEMA, EvidenceClaim, PatientCase, RetrievalHit, SourceChunk
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


def extract_evidence_claims(
    hits: list[RetrievalHit],
    case: PatientCase,
    *,
    extractor: str = "deterministic",
    llm_command: str | None = None,
    llm_timeout_seconds: float = 60.0,
    llm_fallback_to_deterministic: bool = False,
) -> list[EvidenceClaim]:
    if extractor == "deterministic":
        return extract_deterministic_evidence_claims(hits, case)
    if extractor == "llm":
        return extract_llm_evidence_claims(
            hits,
            case,
            command=llm_command,
            timeout_seconds=llm_timeout_seconds,
            fallback_to_deterministic=llm_fallback_to_deterministic,
        )
    raise ValueError(f"unsupported evidence extractor: {extractor}")


def extract_deterministic_evidence_claims(hits: list[RetrievalHit], case: PatientCase) -> list[EvidenceClaim]:
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


def extract_llm_evidence_claims(
    hits: list[RetrievalHit],
    case: PatientCase,
    *,
    command: str | None = None,
    timeout_seconds: float = 60.0,
    fallback_to_deterministic: bool = False,
) -> list[EvidenceClaim]:
    command = command or os.environ.get("MEDREASON_LLM_COMMAND")
    if not command:
        raise ValueError("LLM extraction requires --llm-command or MEDREASON_LLM_COMMAND")

    claims: list[EvidenceClaim] = []
    seen: set[tuple[str, str, str | None, str, str]] = set()
    for hit in hits:
        try:
            raw_items = _run_llm_command(
                command=command,
                payload=build_llm_extraction_payload(hit, case),
                timeout_seconds=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
            if not fallback_to_deterministic:
                raise RuntimeError(f"LLM extraction failed for chunk {hit.chunk.id}: {exc}") from exc
            log_event(logger, "llm_extraction_failed_with_fallback", chunk_id=hit.chunk.id, error=str(exc))
            claims.extend(extract_deterministic_evidence_claims([hit], case))
            continue

        accepted_for_hit = 0
        for item in raw_items:
            claim = _claim_from_llm_item(item, hit.chunk)
            if claim is None:
                log_event(logger, "llm_claim_rejected", chunk_id=hit.chunk.id, reason="unusable_span_or_fields")
                continue
            if _llm_claim_contradicts_span_semantics(claim):
                log_event(logger, "llm_claim_rejected", chunk_id=hit.chunk.id, claim_id=claim.id, reason="span_semantics")
                continue
            if not _llm_claim_relevant_to_case(claim, case):
                log_event(logger, "llm_claim_rejected", chunk_id=hit.chunk.id, claim_id=claim.id, reason="not_patient_grounded")
                continue
            errors = validate_evidence_claim(claim, hit.chunk.text)
            if errors:
                log_event(logger, "llm_claim_rejected", chunk_id=hit.chunk.id, claim_id=claim.id, errors=errors)
                continue
            key = (claim.condition, claim.finding or "", claim.polarity, claim.source_id, claim.sentence)
            if key in seen:
                continue
            seen.add(key)
            accepted_for_hit += 1
            claims.append(claim)
        if fallback_to_deterministic and accepted_for_hit == 0:
            claims.extend(extract_deterministic_evidence_claims([hit], case))

    log_event(logger, "llm_evidence_extracted", hits=len(hits), claims=len(claims), command=_command_label(command))
    return claims


def build_llm_extraction_payload(hit: RetrievalHit, case: PatientCase) -> dict[str, Any]:
    return {
        "task": "extract_medical_evidence_claims",
        "schema": EVIDENCE_CLAIM_SCHEMA,
        "instructions": (
            "Extract only evidence claims directly supported by the source passage. "
            "Return JSON only: {\"claims\": [...]}. Every claim must include claim_type, "
            "condition, finding, polarity, strength, exact_quote, source_span_start, "
            "source_span_end, and extraction_confidence. The exact_quote must be copied "
            "verbatim from source.text. Do not infer facts from outside the source."
        ),
        "allowed_claim_types": EVIDENCE_CLAIM_SCHEMA["properties"]["claim_type"]["enum"],
        "allowed_polarities": EVIDENCE_CLAIM_SCHEMA["properties"]["polarity"]["enum"],
        "allowed_strengths": EVIDENCE_CLAIM_SCHEMA["properties"]["strength"]["enum"],
        "case": case.to_dict(),
        "retrieval": {
            "score": hit.score,
            "matched_terms": hit.matched_terms,
            "score_parts": hit.score_parts,
            "query_labels": hit.query_labels,
        },
        "source": hit.chunk.to_dict(),
    }


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


def _run_llm_command(*, command: str, payload: dict[str, Any], timeout_seconds: float) -> list[dict[str, Any]]:
    completed = subprocess.run(
        shlex.split(command),
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise subprocess.SubprocessError(stderr or f"LLM command exited with {completed.returncode}")
    output = json.loads(completed.stdout)
    if isinstance(output, list):
        raw_claims = output
    elif isinstance(output, dict):
        raw_claims = output.get("claims", [])
    else:
        raise ValueError("LLM command must return a JSON object or array")
    if not isinstance(raw_claims, list):
        raise ValueError("LLM command JSON field 'claims' must be a list")
    return [claim for claim in raw_claims if isinstance(claim, dict)]


def _claim_from_llm_item(item: dict[str, Any], chunk: SourceChunk) -> EvidenceClaim | None:
    span = _resolve_llm_span(item, chunk.text)
    if span is None:
        return None
    sentence, span_start, span_end = span
    claim_type = _normalize_claim_type(str(item.get("claim_type", "")))
    if not claim_type:
        return None
    condition = _canonical_or_raw(str(item.get("condition", "")), kind="condition")
    if not condition:
        return None
    finding_raw = item.get("finding") or item.get("finding_or_test") or item.get("test") or item.get("intervention")
    finding = _canonical_or_raw(str(finding_raw), kind=None) if finding_raw else None
    polarity = _normalize_polarity(str(item.get("polarity", "")), claim_type)
    if polarity == "recommends" and finding and _is_test_concept(finding):
        claim_type = "requires_test"
    strength = _normalize_strength(str(item.get("strength", "")))
    confidence = _normalize_confidence(item.get("extraction_confidence", item.get("confidence", 0.5)))
    claim_id = _claim_id(condition, finding or "", polarity, chunk.id, sentence)
    return EvidenceClaim(
        id=claim_id,
        claim_type=claim_type,
        condition=condition,
        finding=finding,
        polarity=polarity,
        strength=strength,
        source_id=chunk.id,
        source_type=chunk.source_type,
        source_title=chunk.title,
        section_path=chunk.section_path,
        paragraph_index=chunk.paragraph_index,
        sentence=sentence,
        source_span_start=span_start,
        source_span_end=span_end,
        source_text_hash=_source_text_hash(chunk.text),
        extraction_confidence=confidence,
        extraction_method="llm_command_v1",
    )


def _resolve_llm_span(item: dict[str, Any], source_text: str) -> tuple[str, int, int] | None:
    quote = str(item.get("exact_quote") or item.get("sentence") or item.get("source_span_text") or "").strip()
    start = _optional_int(item.get("source_span_start"))
    end = _optional_int(item.get("source_span_end"))
    if start is not None and end is not None and 0 <= start < end <= len(source_text):
        span_text = source_text[start:end]
        if not quote or span_text == quote:
            return span_text, start, end
        return None
    if not quote:
        return None
    start = source_text.find(quote)
    if start < 0:
        return None
    return quote, start, start + len(quote)


def _normalize_claim_type(value: str) -> str:
    normalized = normalize(value).replace(" ", "_")
    aliases = {
        "finding_supports_condition": "supports",
        "finding_argues_against_condition": "argues_against",
        "recommends_test": "requires_test",
        "test_recommended": "requires_test",
        "rule_out": "rules_out",
        "rule_in": "rules_in",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = EVIDENCE_CLAIM_SCHEMA["properties"]["claim_type"]["enum"]
    return normalized if normalized in allowed else ""


def _llm_claim_contradicts_span_semantics(claim: EvidenceClaim) -> bool:
    if claim.polarity != "argues_against" and claim.claim_type != "rules_out":
        return False
    lowered = normalize(claim.sentence)
    return any(cue in lowered for cue in NON_RULE_OUT_CUES)


def _llm_claim_relevant_to_case(claim: EvidenceClaim, case: PatientCase) -> bool:
    if claim.polarity == "recommends" or claim.claim_type in {"requires_test", "treatment_recommends"}:
        return claim.finding is not None and _is_test_concept(claim.finding)
    if not claim.finding:
        return False
    patient_concepts = _case_concepts_by_status(case)
    return claim.finding in patient_concepts["present"]


def _is_test_concept(value: str) -> bool:
    concept = CONCEPTS.get(value)
    return concept is not None and concept.kind == "test"


def _normalize_polarity(value: str, claim_type: str) -> str:
    normalized = normalize(value).replace(" ", "_")
    if normalized in EVIDENCE_CLAIM_SCHEMA["properties"]["polarity"]["enum"]:
        return normalized
    if claim_type in {"argues_against", "rules_out", "contraindicates"}:
        return "argues_against"
    if claim_type in {"requires_test", "treatment_recommends"}:
        return "recommends"
    return "supports"


def _normalize_strength(value: str) -> str:
    normalized = normalize(value)
    if normalized in EVIDENCE_CLAIM_SCHEMA["properties"]["strength"]["enum"]:
        return normalized
    return "moderate"


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.5
    return max(0.0, min(round(confidence, 2), 1.0))


def _canonical_or_raw(value: str, kind: str | None) -> str:
    value = value.strip()
    if not value:
        return ""
    return canonicalize_term(value, kind=kind) or normalize(value)


def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _command_label(command: str) -> str:
    parts = shlex.split(command)
    return parts[0] if parts else command


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
