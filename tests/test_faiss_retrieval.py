from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.ingestion import ingest_path
from medreason_graph.models import PatientCase
from medreason_graph.query import decompose_case_query


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_faiss_available(), "faiss-cpu is not installed")
class FAISSRetrievalTest(unittest.TestCase):
    def test_faiss_retriever_returns_vector_hits(self) -> None:
        from medreason_graph.faiss_retrieval import FAISSRetriever, build_faiss_index

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
            index_path = Path(temp_dir) / "corpus.faiss"
            build_faiss_index(chunks, index_path)
            case = PatientCase.from_dict(
                {
                    "case_id": "case",
                    "patient": {},
                    "chief_complaint": "chest pain",
                    "findings": [{"type": "symptom", "name": "left arm radiation", "status": "present"}],
                }
            )
            retriever = FAISSRetriever(index_path)
            try:
                hits = retriever.fused_search(decompose_case_query(case), top_k=5)
            finally:
                retriever.close()

        self.assertTrue(hits)
        self.assertTrue(any("faiss_ip" in key for hit in hits for key in hit.score_parts))
        self.assertEqual(retriever.embedding_config.preset, "hash")

    def test_analyzer_can_use_faiss_retriever(self) -> None:
        from medreason_graph.faiss_retrieval import FAISSRetriever, build_faiss_index

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "guideline.md"
            source.write_text(
                "# ACS Guideline\n\n"
                "## Symptoms\n\n"
                "Left arm radiation and chest pain support acute coronary syndrome.\n",
                encoding="utf-8",
            )
            chunks = ingest_path(source, source_type="guideline")
            index_path = Path(temp_dir) / "corpus.faiss"
            build_faiss_index(chunks, index_path)
            retriever = FAISSRetriever(index_path)
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
