from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EVIDENCE_CLAIM_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EvidenceClaim",
    "type": "object",
    "required": [
        "id",
        "claim_type",
        "condition",
        "polarity",
        "source_id",
        "source_span_start",
        "source_span_end",
        "source_text_hash",
        "sentence",
        "extraction_confidence",
        "extraction_method",
    ],
    "properties": {
        "schema_version": {"const": "evidence_claim.v1"},
        "id": {"type": "string"},
        "claim_type": {
            "type": "string",
            "enum": [
                "supports",
                "argues_against",
                "requires_test",
                "rules_in",
                "rules_out",
                "contraindicates",
                "red_flag",
                "treatment_recommends",
            ],
        },
        "condition": {"type": "string"},
        "finding": {"type": ["string", "null"]},
        "polarity": {"type": "string", "enum": ["supports", "argues_against", "recommends"]},
        "strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
        "source_id": {"type": "string"},
        "source_span_start": {"type": "integer", "minimum": 0},
        "source_span_end": {"type": "integer", "minimum": 1},
        "source_text_hash": {"type": "string"},
        "sentence": {"type": "string"},
        "extraction_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "extraction_method": {"type": "string"},
    },
    "additionalProperties": True,
}


@dataclass(frozen=True)
class SourceChunk:
    id: str
    source_id: str
    title: str
    source_type: str
    section_path: list[str]
    section_type: str
    paragraph_index: int
    text: str
    publication_date: str | None = None
    authors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceChunk":
        return cls(**data)


@dataclass(frozen=True)
class PatientFinding:
    type: str
    name: str
    status: str = "present"
    value: Any = None
    unit: str | None = None
    concept: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatientFinding":
        return cls(**data)


@dataclass(frozen=True)
class PatientCase:
    case_id: str
    patient: dict[str, Any]
    chief_complaint: str
    findings: list[PatientFinding]
    free_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "patient": self.patient,
            "chief_complaint": self.chief_complaint,
            "findings": [finding.to_dict() for finding in self.findings],
            "free_text": self.free_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatientCase":
        findings = [
            finding if isinstance(finding, PatientFinding) else PatientFinding.from_dict(finding)
            for finding in data.get("findings", [])
        ]
        return cls(
            case_id=data.get("case_id", "case"),
            patient=data.get("patient", {}),
            chief_complaint=data.get("chief_complaint", ""),
            findings=findings,
            free_text=data.get("free_text", ""),
        )


@dataclass(frozen=True)
class RetrievalHit:
    chunk: SourceChunk
    score: float
    matched_terms: list[str]
    score_parts: dict[str, float]
    query_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "matched_terms": self.matched_terms,
            "score_parts": self.score_parts,
            "query_labels": self.query_labels,
        }


@dataclass(frozen=True)
class EvidenceClaim:
    id: str
    claim_type: str
    condition: str
    finding: str | None
    polarity: str
    strength: str
    source_id: str
    source_type: str
    source_title: str
    section_path: list[str]
    paragraph_index: int
    sentence: str
    source_span_start: int = -1
    source_span_end: int = -1
    source_text_hash: str = ""
    extraction_confidence: float = 0.0
    extraction_method: str = "unknown"
    schema_version: str = "evidence_claim.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceClaim":
        return cls(**data)


@dataclass(frozen=True)
class ReasoningStep:
    id: str
    condition: str
    statement: str
    uses_evidence: list[str]
    patient_facts: list[str]
    polarity: str = "supports"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReasoningStep":
        return cls(**data)


@dataclass(frozen=True)
class DifferentialItem:
    condition: str
    rank: int
    urgency: str
    score: float
    evidence_for: list[str]
    evidence_against: list[str]
    missing_evidence: list[str]
    reasoning_steps: list[str]
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DifferentialItem":
        return cls(**data)


@dataclass(frozen=True)
class EvidenceGraph:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceGraph":
        return cls(nodes=data.get("nodes", []), edges=data.get("edges", []))


@dataclass(frozen=True)
class VerifierReport:
    unsupported_claims: list[str]
    source_conflicts: list[str]
    missing_patient_facts: list[str]
    dangerous_misses_checked: list[str]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifierReport":
        return cls(**data)


@dataclass(frozen=True)
class AnalysisResult:
    case_id: str
    problem_representation: str
    differential: list[DifferentialItem]
    evidence_claims: list[EvidenceClaim]
    reasoning_steps: list[ReasoningStep]
    graph: EvidenceGraph
    verifier: VerifierReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "problem_representation": self.problem_representation,
            "differential": [item.to_dict() for item in self.differential],
            "evidence_claims": [claim.to_dict() for claim in self.evidence_claims],
            "reasoning_steps": [step.to_dict() for step in self.reasoning_steps],
            "graph": self.graph.to_dict(),
            "verifier": self.verifier.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisResult":
        return cls(
            case_id=data["case_id"],
            problem_representation=data.get("problem_representation", ""),
            differential=[DifferentialItem.from_dict(item) for item in data.get("differential", [])],
            evidence_claims=[EvidenceClaim.from_dict(item) for item in data.get("evidence_claims", [])],
            reasoning_steps=[ReasoningStep.from_dict(item) for item in data.get("reasoning_steps", [])],
            graph=EvidenceGraph.from_dict(data.get("graph", {})),
            verifier=VerifierReport.from_dict(data.get("verifier", {})),
        )
