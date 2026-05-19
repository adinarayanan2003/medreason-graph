from __future__ import annotations

from medreason_graph.lexicon import DANGEROUS_ALTERNATIVES
from medreason_graph.lexicon import CONCEPTS
from medreason_graph.models import ClaimVerification, EvidenceClaim, PatientCase, ReasoningStep, VerifierReport
from medreason_graph.text import canonicalize_term, detect_concepts, normalize, phrase_in_text

CLAIM_VERIFIER_METHOD = "deterministic_claim_verifier_v1"
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
TEST_CUES = ("obtain", "measure", "test", "evaluate", "initial evaluation", "serial", "perform", "imaging")
RED_FLAG_CUES = ("red flag", "emergent", "emergency", "urgent", "dangerous", "high risk", "concerning")


def verify_evidence_claims(claims: list[EvidenceClaim]) -> list[ClaimVerification]:
    return [_verify_evidence_claim(claim) for claim in claims]


def _verify_evidence_claim(claim: EvidenceClaim) -> ClaimVerification:
    reasons: list[str] = []
    sentence = claim.sentence
    lowered = normalize(sentence)
    source_context = " ".join([claim.source_title, *claim.section_path, sentence])
    sentence_conditions = detect_concepts(sentence, kind="condition")
    context_conditions = detect_concepts(source_context, kind="condition")
    sentence_concepts = detect_concepts(sentence)

    if (
        claim.condition not in sentence_conditions
        and claim.condition not in context_conditions
        and not phrase_in_text(claim.condition, sentence)
        and not phrase_in_text(claim.condition, source_context)
    ):
        reasons.append("condition_not_in_source_context")

    if claim.polarity == "argues_against" or claim.claim_type == "rules_out":
        if claim.condition not in sentence_conditions and not phrase_in_text(claim.condition, sentence):
            reasons.append("negative_claim_condition_not_in_source_span")

    if claim.finding and claim.polarity != "recommends":
        if claim.finding not in sentence_concepts and not phrase_in_text(claim.finding, sentence):
            reasons.append("finding_not_in_source_span")

    if claim.polarity == "supports" and any(cue in lowered for cue in DIFFERENTIAL_ONLY_CUES):
        reasons.append("differential_language_not_support")

    if claim.polarity == "argues_against" and any(cue in lowered for cue in NON_RULE_OUT_CUES):
        reasons.append("negated_rule_out_misread")

    if claim.claim_type == "rules_out" and any(cue in lowered for cue in NON_RULE_OUT_CUES):
        reasons.append("negated_rule_out_misread")

    if claim.claim_type == "requires_test" or claim.polarity == "recommends":
        if not claim.finding or not _is_test_concept(claim.finding):
            reasons.append("recommended_item_not_test")
        if not any(cue in lowered for cue in TEST_CUES):
            reasons.append("missing_test_recommendation_language")

    if claim.claim_type == "red_flag" and not any(cue in lowered for cue in RED_FLAG_CUES):
        reasons.append("missing_red_flag_language")

    if claim.claim_type == "treatment_recommends":
        reasons.append("treatment_recommendations_not_enabled")

    return ClaimVerification(
        claim_id=claim.id,
        supported=not reasons,
        reasons=sorted(set(reasons)),
        claim_type=claim.claim_type,
        condition=claim.condition,
        finding=claim.finding,
        polarity=claim.polarity,
        sentence=claim.sentence,
        source_title=claim.source_title,
        source_span_start=claim.source_span_start,
        source_span_end=claim.source_span_end,
        verifier_method=CLAIM_VERIFIER_METHOD,
    )


def _is_test_concept(value: str) -> bool:
    concept = CONCEPTS.get(value)
    return concept is not None and concept.kind == "test"


def verify_reasoning(
    case: PatientCase,
    claims: list[EvidenceClaim],
    steps: list[ReasoningStep],
    dangerous_checked: list[str],
) -> VerifierReport:
    claim_by_id = {claim.id: claim for claim in claims}
    unsupported_claims: list[str] = []
    source_conflicts: list[str] = []
    missing_patient_facts: list[str] = []

    patient_fact_concepts = _patient_fact_concepts(case)
    for step in steps:
        if not step.uses_evidence:
            unsupported_claims.append(step.id)
            continue
        cited_claims = [claim_by_id.get(claim_id) for claim_id in step.uses_evidence]
        if any(claim is None for claim in cited_claims):
            unsupported_claims.append(step.id)
            continue
        for claim in cited_claims:
            assert claim is not None
            if claim.condition != step.condition:
                source_conflicts.append(step.id)
            if step.polarity == "supports" and claim.polarity not in {"supports", "recommends"}:
                source_conflicts.append(step.id)
            if step.polarity == "argues_against" and claim.polarity != "argues_against":
                source_conflicts.append(step.id)
        for fact in step.patient_facts:
            canonical = canonicalize_term(fact) or fact
            if canonical not in patient_fact_concepts and fact not in {"missing critical test", "dangerous alternative"}:
                missing_patient_facts.append(f"{step.id}:{fact}")

    expected_dangerous = _expected_dangerous(case)
    missing_dangerous = sorted(set(expected_dangerous) - set(dangerous_checked))
    source_conflicts.extend(f"dangerous_miss_not_checked:{condition}" for condition in missing_dangerous)

    passed = not unsupported_claims and not source_conflicts and not missing_patient_facts
    return VerifierReport(
        unsupported_claims=sorted(set(unsupported_claims)),
        source_conflicts=sorted(set(source_conflicts)),
        missing_patient_facts=sorted(set(missing_patient_facts)),
        dangerous_misses_checked=sorted(set(dangerous_checked)),
        passed=passed,
    )


def _patient_fact_concepts(case: PatientCase) -> set[str]:
    concepts = detect_concepts(case.chief_complaint)
    concepts.update(detect_concepts(case.free_text))
    context = " ".join([case.chief_complaint, case.free_text, " ".join(finding.name for finding in case.findings)])
    for finding in case.findings:
        canonical = finding.concept or canonicalize_term(finding.name, context=context)
        if canonical:
            concepts.add(canonical)
    return concepts


def _expected_dangerous(case: PatientCase) -> list[str]:
    concepts = detect_concepts(case.chief_complaint) | detect_concepts(case.free_text)
    context = " ".join([case.chief_complaint, case.free_text, " ".join(finding.name for finding in case.findings)])
    for finding in case.findings:
        canonical = finding.concept or canonicalize_term(finding.name, context=context)
        if canonical:
            concepts.add(canonical)
    expected: list[str] = []
    for concept in concepts:
        expected.extend(DANGEROUS_ALTERNATIVES.get(concept, ()))
    return sorted(set(expected))
