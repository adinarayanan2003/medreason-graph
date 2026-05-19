from __future__ import annotations

from medreason_graph.models import AnalysisResult


def render_text(result: AnalysisResult) -> str:
    lines = [
        f"Case: {result.case_id}",
        f"Problem: {result.problem_representation}",
        f"Verifier: {'PASS' if result.verifier.passed else 'FAIL'}",
        "",
        "Differential:",
    ]
    claims_by_id = {claim.id: claim for claim in result.evidence_claims}
    steps_by_id = {step.id: step for step in result.reasoning_steps}
    for item in result.differential:
        lines.append(
            f"{item.rank}. {item.condition} [{item.urgency}] score={item.score} confidence={item.confidence}"
        )
        if item.missing_evidence:
            lines.append(f"   Missing evidence: {', '.join(item.missing_evidence)}")
        shown_steps = 0
        shown_statements: set[str] = set()
        for step_id in item.reasoning_steps:
            if shown_steps >= 5:
                break
            step = steps_by_id[step_id]
            if step.statement in shown_statements:
                continue
            shown_statements.add(step.statement)
            shown_steps += 1
            sources = ", ".join(_source_label(claims_by_id[claim_id]) for claim_id in step.uses_evidence if claim_id in claims_by_id)
            lines.append(f"   - {step.statement} ({sources})")
    if result.verifier.source_conflicts or result.verifier.unsupported_claims:
        lines.extend(["", "Verifier findings:"])
        for claim in result.verifier.unsupported_claims:
            lines.append(f"   Unsupported reasoning step: {claim}")
        for conflict in result.verifier.source_conflicts:
            lines.append(f"   Source conflict: {conflict}")
    if result.verifier.dangerous_misses_checked:
        lines.extend(["", f"Dangerous misses checked: {', '.join(result.verifier.dangerous_misses_checked)}"])
    return "\n".join(lines)


def _source_label(claim) -> str:
    section_path = claim.section_path[1:] if claim.section_path and claim.section_path[0] == claim.source_title else claim.section_path
    section = " > ".join(section_path[-2:] or claim.section_path[-1:])
    return f"{claim.source_title}, {section}, paragraph {claim.paragraph_index}"
