from __future__ import annotations

import json

from medreason_graph.models import EvidenceClaim, EvidenceGraph, PatientCase, ReasoningStep
from medreason_graph.text import canonicalize_term, detect_concepts


def build_graph(case: PatientCase, claims: list[EvidenceClaim], steps: list[ReasoningStep]) -> EvidenceGraph:
    nodes: dict[str, dict[str, object]] = {}
    edges: list[dict[str, str]] = []

    def add_node(node_id: str, kind: str, label: str, **attrs: object) -> None:
        nodes.setdefault(node_id, {"id": node_id, "kind": kind, "label": label, **attrs})

    add_node(f"case:{case.case_id}", "PatientCase", case.case_id)
    for concept in detect_concepts(case.chief_complaint) | detect_concepts(case.free_text):
        add_node(f"concept:{concept}", "MedicalConcept", concept)
        edges.append({"source": f"case:{case.case_id}", "type": "mentions", "target": f"concept:{concept}"})

    for finding in case.findings:
        fact_id = f"fact:{finding.name}:{finding.status}"
        add_node(fact_id, "PatientFinding", finding.name, status=finding.status, finding_type=finding.type)
        edges.append({"source": f"case:{case.case_id}", "type": "has_finding", "target": fact_id})
        canonical = finding.concept or canonicalize_term(finding.name)
        if canonical:
            add_node(f"concept:{canonical}", "MedicalConcept", canonical)
            edges.append({"source": fact_id, "type": "maps_to", "target": f"concept:{canonical}"})

    for claim in claims:
        add_node(f"condition:{claim.condition}", "Condition", claim.condition)
        add_node(
            f"evidence:{claim.id}",
            "EvidenceClaim",
            claim.id,
            claim_type=claim.claim_type,
            polarity=claim.polarity,
            strength=claim.strength,
            extraction_confidence=claim.extraction_confidence,
            extraction_method=claim.extraction_method,
        )
        add_node(
            f"source:{claim.source_id}",
            "SourcePassage",
            claim.source_title,
            section_path=claim.section_path,
            span_start=claim.source_span_start,
            span_end=claim.source_span_end,
            source_text_hash=claim.source_text_hash,
        )
        if claim.finding:
            add_node(f"concept:{claim.finding}", "MedicalConcept", claim.finding)
            edges.append({"source": f"evidence:{claim.id}", "type": "about_finding", "target": f"concept:{claim.finding}"})
        edge_type = {
            "supports": "supports",
            "argues_against": "argues_against",
            "recommends": "recommends_test_for",
        }.get(claim.polarity, claim.polarity)
        edges.append({"source": f"evidence:{claim.id}", "type": edge_type, "target": f"condition:{claim.condition}"})
        edges.append({"source": f"evidence:{claim.id}", "type": "has_source", "target": f"source:{claim.source_id}"})

    for step in steps:
        add_node(f"reasoning:{step.id}", "ReasoningStep", step.statement, condition=step.condition)
        add_node(f"condition:{step.condition}", "Condition", step.condition)
        edges.append({"source": f"reasoning:{step.id}", "type": "concludes", "target": f"condition:{step.condition}"})
        for claim_id in step.uses_evidence:
            edges.append({"source": f"reasoning:{step.id}", "type": "uses_evidence", "target": f"evidence:{claim_id}"})

    return EvidenceGraph(nodes=list(nodes.values()), edges=edges)


def export_cytoscape(graph: EvidenceGraph) -> dict[str, list[dict[str, object]]]:
    elements: list[dict[str, object]] = []
    for node in graph.nodes:
        elements.append({"data": dict(node)})
    for index, edge in enumerate(graph.edges):
        elements.append(
            {
                "data": {
                    "id": f"edge_{index}",
                    "source": edge["source"],
                    "target": edge["target"],
                    "label": edge["type"],
                }
            }
        )
    return {"elements": elements}


def export_graphviz_dot(graph: EvidenceGraph) -> str:
    lines = ["digraph MedReasonGraph {", '  rankdir="LR";']
    for node in graph.nodes:
        node_id = _dot_id(str(node["id"]))
        label = _dot_label(f'{node["kind"]}: {node["label"]}')
        lines.append(f'  {node_id} [label="{label}"];')
    for edge in graph.edges:
        source = _dot_id(edge["source"])
        target = _dot_id(edge["target"])
        label = _dot_label(edge["type"])
        lines.append(f'  {source} -> {target} [label="{label}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def graph_to_json(graph: EvidenceGraph) -> str:
    return json.dumps(graph.to_dict(), indent=2)


def _dot_id(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in value)
    return f"n_{safe}"


def _dot_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
