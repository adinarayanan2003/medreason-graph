from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.graph_store import (
    build_graph_store,
    query_evidence,
    query_explain_rank,
    query_missing_tests,
    query_reasoning,
    query_source_spans,
    query_verifier_failures,
)
from medreason_graph.ingestion import ingest_path
from medreason_graph.models import PatientCase, SourceChunk

ROOT = Path(__file__).resolve().parents[1]


class GraphStoreTest(unittest.TestCase):
    def test_persisted_graph_answers_condition_queries(self) -> None:
        chunks = ingest_path(ROOT / "examples" / "corpus", source_type="guideline")
        case = PatientCase.from_dict(
            {
                "case_id": "case_graph_store",
                "patient": {"age": 45, "sex": "male"},
                "chief_complaint": "crushing chest pain",
                "findings": [
                    {"type": "symptom", "name": "left arm radiation", "status": "present"},
                    {"type": "symptom", "name": "diaphoresis", "status": "present"},
                    {"type": "symptom", "name": "nausea", "status": "present"},
                    {"type": "test", "name": "ECG", "status": "missing"},
                    {"type": "test", "name": "troponin", "status": "missing"},
                ],
                "free_text": "45M with crushing chest pain, sweating, nausea, and left arm radiation.",
            }
        )
        result = MedReasonAnalyzer(chunks).analyze(case)

        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "case_graph.sqlite"
            build_graph_store(result, graph_path)

            evidence_for = query_evidence(graph_path, condition="acute coronary syndrome", polarity="supports")
            evidence_against = query_evidence(
                graph_path,
                condition="gastroesophageal reflux disease",
                polarity="argues_against",
            )
            missing_tests = query_missing_tests(graph_path, condition="acute coronary syndrome")
            reasoning = query_reasoning(graph_path, condition="acute coronary syndrome")
            source_spans = query_source_spans(graph_path, condition="acute coronary syndrome")
            explanation = query_explain_rank(graph_path, condition="acute coronary syndrome")
            failures = query_verifier_failures(graph_path)

        self.assertTrue(evidence_for)
        self.assertTrue(evidence_against)
        self.assertIn("ecg", {item["test_name"] for item in missing_tests})
        self.assertTrue(all(step["uses_evidence"] for step in reasoning))
        self.assertTrue(all("source_span_start" in span for span in source_spans))
        self.assertTrue(explanation["found"])
        self.assertEqual(explanation["condition"], "acute coronary syndrome")
        self.assertTrue(explanation["evidence_for"])
        self.assertEqual(failures, [])

    def test_verifier_failures_include_rejected_claim_snapshot(self) -> None:
        case = PatientCase.from_dict(
            {
                "case_id": "case_graph_failures",
                "patient": {},
                "chief_complaint": "dyspnea",
                "findings": [{"type": "symptom", "name": "shortness of breath", "status": "present"}],
            }
        )
        chunk = SourceChunk(
            id="chunk",
            source_id="source",
            title="Acute Pulmonary Embolism - StatPearls",
            source_type="textbook",
            section_path=["Acute Pulmonary Embolism - StatPearls", "Evaluation"],
            section_type="diagnostic_criteria",
            paragraph_index=1,
            text="It helps to rule out alternative diagnoses in patients presenting with acute dyspnea.",
        )
        result = MedReasonAnalyzer([chunk]).analyze(case)

        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "case_graph.sqlite"
            build_graph_store(result, graph_path)
            failures = query_verifier_failures(graph_path)

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["condition"], "pulmonary embolism")
        self.assertEqual(failures[0]["finding"], "dyspnea")
        self.assertIn("negative_claim_condition_not_in_source_span", failures[0]["reasons"])


if __name__ == "__main__":
    unittest.main()
