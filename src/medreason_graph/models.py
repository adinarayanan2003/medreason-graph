from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


@dataclass(frozen=True)
class EvidenceGraph:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}


@dataclass(frozen=True)
class VerifierReport:
    unsupported_claims: list[str]
    source_conflicts: list[str]
    missing_patient_facts: list[str]
    dangerous_misses_checked: list[str]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
