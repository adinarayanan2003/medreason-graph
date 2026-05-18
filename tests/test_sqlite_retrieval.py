from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.ingestion import ingest_path
from medreason_graph.models import PatientCase
from medreason_graph.query import decompose_case_query
from medreason_graph.sqlite_retrieval import SQLiteFTSRetriever, build_sqlite_fts_index


class SQLiteRetrievalTest(unittest.TestCase):
    def test_sqlite_fts_retriever_returns_medical_hits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "guideline.md"
            source.write_text(
                "# ACS Guideline\n\n"
                "## Symptoms\n\n"
                "Chest pain radiating to the left arm supports acute coronary syndrome.\n\n"
                "## Diagnostic Tests\n\n"
                "ECG and troponin should be obtained for acute coronary syndrome.\n",
                encoding="utf-8",
            )
            chunks = ingest_path(source, source_type="guideline")
            index_path = Path(temp_dir) / "corpus.sqlite"
            build_sqlite_fts_index(chunks, index_path)
            case = PatientCase.from_dict(
                {
                    "case_id": "case",
                    "patient": {},
                    "chief_complaint": "chest pain",
                    "findings": [
                        {"type": "symptom", "name": "left arm radiation", "status": "present"},
                        {"type": "test", "name": "ECG", "status": "missing"},
                    ],
                }
            )
            retriever = SQLiteFTSRetriever(index_path)
            try:
                hits = retriever.fused_search(decompose_case_query(case), top_k=5)
            finally:
                retriever.close()

        self.assertTrue(hits)
        self.assertTrue(any(hit.chunk.section_type in {"symptoms", "diagnostic_criteria"} for hit in hits))
        self.assertTrue(any("fts_bm25" in key for hit in hits for key in hit.score_parts))

    def test_analyzer_can_use_sqlite_retriever(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "guideline.md"
            source.write_text(
                "# ACS Guideline\n\n"
                "## Symptoms\n\n"
                "Left arm radiation and chest pain support acute coronary syndrome.\n",
                encoding="utf-8",
            )
            chunks = ingest_path(source, source_type="guideline")
            index_path = Path(temp_dir) / "corpus.sqlite"
            build_sqlite_fts_index(chunks, index_path)
            retriever = SQLiteFTSRetriever(index_path)
            try:
                result = MedReasonAnalyzer(chunks, retriever=retriever).analyze(
                    {
                        "case_id": "case",
                        "patient": {},
                        "chief_complaint": "chest pain",
                        "findings": [{"type": "symptom", "name": "left arm radiation", "status": "present"}],
                    }
                )
            finally:
                retriever.close()

        self.assertTrue(result.verifier.passed)
        self.assertEqual(result.differential[0].condition, "acute coronary syndrome")


if __name__ == "__main__":
    unittest.main()

