from __future__ import annotations

import hashlib
import logging

from medreason_graph.evidence import extract_evidence_claims
from medreason_graph.graph import build_graph
from medreason_graph.lexicon import DANGEROUS_ALTERNATIVES, HIGH_RISK_CONDITIONS, SOURCE_TYPE_WEIGHT, STRENGTH_WEIGHT
from medreason_graph.logging_utils import log_event
from medreason_graph.models import (
    AnalysisResult,
    DifferentialItem,
    EvidenceClaim,
    PatientCase,
    ReasoningStep,
    SourceChunk,
)
from medreason_graph.query import decompose_case_query
from medreason_graph.retrieval_backend import RetrievalBackend
from medreason_graph.retrieval import HybridRetriever
from medreason_graph.text import canonicalize_term, detect_concepts
from medreason_graph.verifier import verify_reasoning

logger = logging.getLogger(__name__)


class MedReasonAnalyzer:
    def __init__(
        self,
        chunks: list[SourceChunk],
        retriever: RetrievalBackend | None = None,
        *,
        evidence_extractor: str = "deterministic",
        llm_command: str | None = None,
        llm_timeout_seconds: float = 60.0,
        llm_fallback_to_deterministic: bool = False,
    ):
        self.chunks = chunks
        self.retriever = retriever or HybridRetriever(chunks)
        self.evidence_extractor = evidence_extractor
        self.llm_command = llm_command
        self.llm_timeout_seconds = llm_timeout_seconds
        self.llm_fallback_to_deterministic = llm_fallback_to_deterministic

    def analyze(self, case: PatientCase | dict, top_k: int = 16) -> AnalysisResult:
        patient_case = case if isinstance(case, PatientCase) else PatientCase.from_dict(case)
        normalized_case = _normalize_case(patient_case)
        query_parts = decompose_case_query(normalized_case)
        query = " | ".join(f"{part.label}:{part.text}" for part in query_parts)
        log_event(logger, "analysis_started", case_id=normalized_case.case_id, query=query, chunks=len(self.chunks))
        hits = self.retriever.fused_search(query_parts, top_k=top_k)
        claims = extract_evidence_claims(
            hits,
            normalized_case,
            extractor=self.evidence_extractor,
            llm_command=self.llm_command,
            llm_timeout_seconds=self.llm_timeout_seconds,
            llm_fallback_to_deterministic=self.llm_fallback_to_deterministic,
        )
        dangerous_checked = _dangerous_alternatives(normalized_case)
        candidates = sorted(_candidate_conditions(claims, dangerous_checked))
        steps = _reasoning_steps(normalized_case, candidates, claims)
        differential = _rank_differential(candidates, claims, steps, normalized_case)
        graph = build_graph(normalized_case, claims, steps)
        verifier = verify_reasoning(normalized_case, claims, steps, dangerous_checked)
        log_event(
            logger,
            "analysis_completed",
            case_id=normalized_case.case_id,
            claims=len(claims),
            reasoning_steps=len(steps),
            differential_items=len(differential),
            verifier_passed=verifier.passed,
        )
        return AnalysisResult(
            case_id=normalized_case.case_id,
            problem_representation=_problem_representation(normalized_case),
            differential=differential,
            evidence_claims=claims,
            reasoning_steps=steps,
            graph=graph,
            verifier=verifier,
        )


def _normalize_case(case: PatientCase) -> PatientCase:
    findings = []
    context = " ".join([case.chief_complaint, case.free_text, " ".join(finding.name for finding in case.findings)])
    for finding in case.findings:
        concept = finding.concept or canonicalize_term(finding.name, kind=finding.type, context=context) or canonicalize_term(finding.name, context=context)
        findings.append(
            type(finding)(
                type=finding.type,
                name=finding.name,
                status=finding.status,
                value=finding.value,
                unit=finding.unit,
                concept=concept,
            )
        )
    return PatientCase(
        case_id=case.case_id,
        patient=case.patient,
        chief_complaint=case.chief_complaint,
        findings=findings,
        free_text=case.free_text,
    )


def _query_from_case(case: PatientCase) -> str:
    present = " ".join(finding.name for finding in case.findings if finding.status == "present")
    missing = " ".join(finding.name for finding in case.findings if finding.status == "missing")
    dangerous = " ".join(_dangerous_alternatives(case))
    return " ".join([case.chief_complaint, case.free_text, present, missing, dangerous]).strip()


def _dangerous_alternatives(case: PatientCase) -> list[str]:
    concepts = detect_concepts(case.chief_complaint) | detect_concepts(case.free_text)
    for finding in case.findings:
        if finding.concept:
            concepts.add(finding.concept)
    dangerous: list[str] = []
    for concept in concepts:
        dangerous.extend(DANGEROUS_ALTERNATIVES.get(concept, ()))
    return sorted(set(dangerous))


def _candidate_conditions(claims: list[EvidenceClaim], dangerous: list[str]) -> set[str]:
    conditions = {claim.condition for claim in claims}
    conditions.update(dangerous)
    return conditions


def _rank_differential(
    candidates: list[str],
    claims: list[EvidenceClaim],
    steps: list[ReasoningStep],
    case: PatientCase,
) -> list[DifferentialItem]:
    by_condition = {condition: [claim for claim in claims if claim.condition == condition] for condition in candidates}
    step_ids = _step_ids_by_condition(steps)
    scored: list[tuple[float, DifferentialItem]] = []
    for condition in candidates:
        condition_claims = by_condition.get(condition, [])
        evidence_for = [claim.id for claim in condition_claims if claim.polarity == "supports"]
        evidence_against = [claim.id for claim in condition_claims if claim.polarity == "argues_against"]
        recommended_tests = [claim.finding for claim in condition_claims if claim.polarity == "recommends" and claim.finding]
        missing = _missing_tests(case, recommended_tests)
        score = _score_condition(condition, condition_claims, bool(missing))
        confidence = _confidence(evidence_for, evidence_against, missing)
        item = DifferentialItem(
            condition=condition,
            rank=0,
            urgency=HIGH_RISK_CONDITIONS.get(condition, "routine"),
            score=round(score, 4),
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            missing_evidence=missing,
            reasoning_steps=step_ids.get(condition, []),
            confidence=confidence,
        )
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    ranked: list[DifferentialItem] = []
    for rank, (_, item) in enumerate(scored, start=1):
        ranked.append(
            DifferentialItem(
                condition=item.condition,
                rank=rank,
                urgency=item.urgency,
                score=item.score,
                evidence_for=item.evidence_for,
                evidence_against=item.evidence_against,
                missing_evidence=item.missing_evidence,
                reasoning_steps=item.reasoning_steps,
                confidence=item.confidence,
            )
        )
    return ranked


def _score_condition(condition: str, claims: list[EvidenceClaim], has_missing_tests: bool) -> float:
    score = 0.0
    support_findings_seen: set[str | None] = set()
    against_findings_seen: set[str | None] = set()
    for claim in claims:
        source_weight = SOURCE_TYPE_WEIGHT.get(claim.source_type, SOURCE_TYPE_WEIGHT["unknown"])
        strength_weight = STRENGTH_WEIGHT.get(claim.strength, 0.5)
        value = source_weight * strength_weight
        if claim.polarity == "supports":
            if claim.finding in support_findings_seen:
                value *= 0.25
            support_findings_seen.add(claim.finding)
            score += 2.5 * value
        elif claim.polarity == "argues_against":
            if claim.finding in against_findings_seen:
                value *= 0.25
            against_findings_seen.add(claim.finding)
            score -= 1.8 * value
        elif claim.polarity == "recommends":
            score += 0.35 * value
    if condition in HIGH_RISK_CONDITIONS:
        score += 0.6
    if has_missing_tests:
        score -= 0.1
    return max(score, 0.0)


def _reasoning_steps(case: PatientCase, candidates: list[str], claims: list[EvidenceClaim]) -> list[ReasoningStep]:
    steps: list[ReasoningStep] = []
    for condition in candidates:
        condition_claims = [claim for claim in claims if claim.condition == condition]
        for claim in condition_claims:
            if claim.polarity == "supports" and claim.finding:
                steps.append(
                    _step(
                        condition=condition,
                        statement=f"{claim.finding} supports considering {condition}.",
                        evidence_ids=[claim.id],
                        patient_facts=[claim.finding],
                        polarity="supports",
                    )
                )
            elif claim.polarity == "argues_against" and claim.finding:
                steps.append(
                    _step(
                        condition=condition,
                        statement=f"{claim.finding} argues against {condition}.",
                        evidence_ids=[claim.id],
                        patient_facts=[claim.finding],
                        polarity="argues_against",
                    )
                )
            elif claim.polarity == "recommends" and claim.finding:
                if claim.finding in _missing_tests(case, [claim.finding]):
                    steps.append(
                        _step(
                            condition=condition,
                            statement=f"{claim.finding} is missing and is a cited diagnostic evidence need for {condition}.",
                            evidence_ids=[claim.id],
                            patient_facts=["missing critical test"],
                            polarity="supports",
                        )
                    )
    return _dedupe_steps(steps)


def _step(
    *,
    condition: str,
    statement: str,
    evidence_ids: list[str],
    patient_facts: list[str],
    polarity: str,
) -> ReasoningStep:
    digest = hashlib.sha1((condition + statement + ",".join(evidence_ids)).encode("utf-8")).hexdigest()[:12]
    return ReasoningStep(
        id=f"rs_{digest}",
        condition=condition,
        statement=statement,
        uses_evidence=evidence_ids,
        patient_facts=patient_facts,
        polarity=polarity,
    )


def _dedupe_steps(steps: list[ReasoningStep]) -> list[ReasoningStep]:
    seen: set[str] = set()
    deduped: list[ReasoningStep] = []
    for step in steps:
        key = f"{step.condition}:{step.statement}:{','.join(step.uses_evidence)}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(step)
    return deduped


def _step_ids_by_condition(steps: list[ReasoningStep]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for step in steps:
        result.setdefault(step.condition, []).append(step.id)
    return result


def _missing_tests(case: PatientCase, tests: list[str | None]) -> list[str]:
    known = {finding.concept or canonicalize_term(finding.name) or finding.name for finding in case.findings}
    missing = {
        test
        for test in tests
        if test and test not in known
    }
    explicit_missing = {
        finding.concept or canonicalize_term(finding.name) or finding.name
        for finding in case.findings
        if finding.status == "missing"
    }
    return sorted(missing | (set(tests) & explicit_missing))


def _confidence(evidence_for: list[str], evidence_against: list[str], missing: list[str]) -> str:
    if len(evidence_for) >= 3 and not evidence_against and not missing:
        return "high"
    if evidence_for and len(missing) <= 2:
        return "moderate"
    return "low"


def _problem_representation(case: PatientCase) -> str:
    age = case.patient.get("age")
    sex = case.patient.get("sex")
    identity = "patient"
    if age and sex:
        identity = f"{age}-year-old {sex}"
    present = [finding.name for finding in case.findings if finding.status == "present"]
    if present:
        return f"{identity} with {case.chief_complaint} and {', '.join(present)}."
    return f"{identity} with {case.chief_complaint}."
