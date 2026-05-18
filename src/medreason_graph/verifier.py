from __future__ import annotations

from medreason_graph.lexicon import DANGEROUS_ALTERNATIVES
from medreason_graph.models import EvidenceClaim, PatientCase, ReasoningStep, VerifierReport
from medreason_graph.text import canonicalize_term, detect_concepts


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
