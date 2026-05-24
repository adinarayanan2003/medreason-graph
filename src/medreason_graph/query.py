from __future__ import annotations

from dataclasses import dataclass, field

from medreason_graph.lexicon import DANGEROUS_ALTERNATIVES
from medreason_graph.models import PatientCase
from medreason_graph.text import canonicalize_term, detect_concepts


@dataclass(frozen=True)
class QueryPart:
    label: str
    text: str
    weight: float
    section_boosts: dict[str, float]
    condition_boosts: set[str]
    presentation_boosts: set[str] = field(default_factory=set)
    condition_tag_boosts: set[str] = field(default_factory=set)
    source_pack_boosts: set[str] = field(default_factory=set)


PRESENTATION_TAGS_BY_CONCEPT = {
    "chest pain": "chest_pain",
    "dyspnea": "dyspnea",
    "headache": "headache",
    "fever": "fever",
}


def decompose_case_query(case: PatientCase) -> list[QueryPart]:
    present = [finding for finding in case.findings if finding.status == "present"]
    absent = [finding for finding in case.findings if finding.status == "absent"]
    missing = [finding for finding in case.findings if finding.status == "missing"]
    dangerous = _dangerous_alternatives(case)
    presentation_tags = _presentation_tags(case)
    parts: list[QueryPart] = []

    base_text = " ".join([case.chief_complaint, case.free_text]).strip()
    if base_text:
        parts.append(
            QueryPart(
                label="case_summary",
                text=base_text,
                weight=1.0,
                section_boosts={"symptoms": 0.35, "diagnostic_criteria": 0.2, "red_flags": 0.25, "differential": 0.2},
                condition_boosts=set(dangerous),
                presentation_boosts=set(presentation_tags),
                condition_tag_boosts=set(dangerous),
                source_pack_boosts=set(presentation_tags),
            )
        )
    if present:
        parts.append(
            QueryPart(
                label="present_findings",
                text=" ".join(finding.name for finding in present),
                weight=1.15,
                section_boosts={"symptoms": 0.45, "physical_exam": 0.2, "diagnostic_criteria": 0.15},
                condition_boosts=set(dangerous),
                presentation_boosts=set(presentation_tags),
                condition_tag_boosts=set(dangerous),
                source_pack_boosts=set(presentation_tags),
            )
        )
    if absent:
        parts.append(
            QueryPart(
                label="absent_findings",
                text=" ".join(finding.name for finding in absent),
                weight=0.5,
                section_boosts={"symptoms": 0.2, "differential": 0.25},
                condition_boosts=set(),
                presentation_boosts=set(presentation_tags),
                source_pack_boosts=set(presentation_tags),
            )
        )
    if missing:
        parts.append(
            QueryPart(
                label="missing_tests",
                text=" ".join(finding.name for finding in missing),
                weight=0.85,
                section_boosts={"diagnostic_criteria": 0.35, "tests": 0.45, "red_flags": 0.15},
                condition_boosts=set(dangerous),
                presentation_boosts=set(presentation_tags),
                condition_tag_boosts=set(dangerous),
                source_pack_boosts=set(presentation_tags),
            )
        )
    if dangerous:
        parts.append(
            QueryPart(
                label="dangerous_alternatives",
                text=" ".join(dangerous),
                weight=0.95,
                section_boosts={"red_flags": 0.45, "differential": 0.35, "diagnostic_criteria": 0.25},
                condition_boosts=set(dangerous),
                presentation_boosts=set(presentation_tags),
                condition_tag_boosts=set(dangerous),
                source_pack_boosts=set(presentation_tags),
            )
        )
    return parts


def _presentation_tags(case: PatientCase) -> list[str]:
    concepts = detect_concepts(case.chief_complaint) | detect_concepts(case.free_text)
    context = " ".join([case.chief_complaint, case.free_text, " ".join(finding.name for finding in case.findings)])
    for finding in case.findings:
        canonical = finding.concept or canonicalize_term(finding.name, context=context)
        if canonical:
            concepts.add(canonical)
    return sorted({PRESENTATION_TAGS_BY_CONCEPT[concept] for concept in concepts if concept in PRESENTATION_TAGS_BY_CONCEPT})


def _dangerous_alternatives(case: PatientCase) -> list[str]:
    concepts = detect_concepts(case.chief_complaint) | detect_concepts(case.free_text)
    context = " ".join([case.chief_complaint, case.free_text, " ".join(finding.name for finding in case.findings)])
    for finding in case.findings:
        canonical = finding.concept or canonicalize_term(finding.name, context=context)
        if canonical:
            concepts.add(canonical)
    dangerous: list[str] = []
    for concept in concepts:
        dangerous.extend(DANGEROUS_ALTERNATIVES.get(concept, ()))
    return sorted(set(dangerous))
