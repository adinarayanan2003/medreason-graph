from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.models import PatientCase, SourceChunk


def evaluate_retrieval(chunks: list[SourceChunk], cases_path: str | Path, *, k: int = 5, retriever=None) -> dict[str, Any]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    analyzer = MedReasonAnalyzer(chunks, retriever=retriever)
    rows: list[dict[str, Any]] = []
    recall_hits = 0
    reciprocal_ranks: list[float] = []

    for item in cases:
        case = PatientCase.from_dict(item["case"])
        expected = item["expected_conditions"]
        result = analyzer.analyze(case, top_k=24)
        ranked_conditions = [entry.condition for entry in result.differential]
        top_k_conditions = ranked_conditions[:k]
        found = [condition for condition in expected if condition in top_k_conditions]
        recall_hits += len(found)
        first_rank = _first_rank(ranked_conditions, expected)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        rows.append(
            {
                "case_id": item["case_id"],
                "expected_conditions": expected,
                "top_conditions": top_k_conditions,
                "found": found,
                "first_relevant_rank": first_rank,
                "verifier_passed": result.verifier.passed,
            }
        )

    expected_total = sum(len(item["expected_conditions"]) for item in cases) or 1
    return {
        "cases": len(cases),
        "k": k,
        "recall_at_k": round(recall_hits / expected_total, 4),
        "mrr": round(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1), 4),
        "rows": rows,
    }


def _first_rank(ranked_conditions: list[str], expected: list[str]) -> int | None:
    expected_set = set(expected)
    for index, condition in enumerate(ranked_conditions, start=1):
        if condition in expected_set:
            return index
    return None
