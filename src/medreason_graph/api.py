from __future__ import annotations

from pathlib import Path
from typing import Any

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.config import load_and_apply_config
from medreason_graph.graph import export_cytoscape, export_graphviz_dot
from medreason_graph.ingestion import ingest_path
from medreason_graph.models import EvidenceGraph, PatientCase, SourceChunk
from medreason_graph.storage import load_chunks, save_chunks

_CASES: dict[str, dict[str, Any]] = {}
_CHUNKS: list[SourceChunk] = []


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError('Install API dependencies with: pip install -e ".[api]"') from exc

    app = FastAPI(title="MedReason Graph", version="0.1.0")

    @app.post("/sources/ingest")
    def ingest_source(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("config_path"):
            load_and_apply_config(payload["config_path"])
        path = payload.get("path")
        if not path:
            raise HTTPException(status_code=400, detail="path is required")
        source_type = payload.get("source_type", "unknown")
        out = payload.get("out")
        chunks = ingest_path(path, source_type=source_type)
        global _CHUNKS
        _CHUNKS = chunks
        if out:
            save_chunks(chunks, out)
        return {"chunks": len(chunks), "out": out}

    @app.post("/cases/analyze")
    def analyze_case(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("config_path"):
            load_and_apply_config(payload["config_path"])
        corpus_path = payload.pop("corpus_path", None)
        evidence_extractor = payload.pop("evidence_extractor", "deterministic")
        llm_command = payload.pop("llm_command", None)
        llm_timeout_seconds = float(payload.pop("llm_timeout_seconds", 60.0))
        llm_fallback_to_deterministic = bool(payload.pop("llm_fallback_to_deterministic", False))
        payload.pop("config_path", None)
        chunks = load_chunks(corpus_path) if corpus_path else _CHUNKS
        if not chunks:
            raise HTTPException(status_code=400, detail="ingest sources or provide corpus_path first")
        case = PatientCase.from_dict(payload)
        result = MedReasonAnalyzer(
            chunks,
            evidence_extractor=evidence_extractor,
            llm_command=llm_command,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_fallback_to_deterministic=llm_fallback_to_deterministic,
        ).analyze(case).to_dict()
        _CASES[case.case_id] = result
        return result

    @app.get("/cases/{case_id}/graph")
    def get_graph(case_id: str, format: str = "json") -> dict[str, Any] | str:
        result = _case_or_404(case_id, HTTPException)
        graph = result["graph"]
        evidence_graph = EvidenceGraph(nodes=graph["nodes"], edges=graph["edges"])
        if format == "cytoscape":
            return export_cytoscape(evidence_graph)
        if format == "dot":
            return export_graphviz_dot(evidence_graph)
        return graph

    @app.get("/cases/{case_id}/evidence")
    def get_evidence(case_id: str) -> list[dict[str, Any]]:
        result = _case_or_404(case_id, HTTPException)
        return result["evidence_claims"]

    @app.get("/cases/{case_id}/audit")
    def get_audit(case_id: str) -> dict[str, Any]:
        result = _case_or_404(case_id, HTTPException)
        return result["verifier"]

    return app


def _case_or_404(case_id: str, http_exception) -> dict[str, Any]:
    if case_id not in _CASES:
        raise http_exception(status_code=404, detail=f"case not found: {case_id}")
    return _CASES[case_id]
