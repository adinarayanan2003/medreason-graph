from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medreason_graph.ingestion import ingest_path
from medreason_graph.retrieval import HybridRetriever


class IngestionRetrievalTest(unittest.TestCase):
    def test_ingestion_preserves_section_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "guideline.md"
            source.write_text(
                "# Chest Pain Guideline\n\n"
                "## Diagnostic Tests\n\n"
                "Initial evaluation should obtain ECG and troponin for acute coronary syndrome.\n",
                encoding="utf-8",
            )
            chunks = ingest_path(source, source_type="guideline")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_type, "guideline")
        self.assertEqual(chunks[0].section_type, "diagnostic_criteria")
        self.assertIn("Diagnostic Tests", chunks[0].section_path)

    def test_retrieval_expands_medical_synonyms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "guideline.md"
            source.write_text(
                "# ACS Guideline\n\n"
                "Acute coronary syndrome includes myocardial infarction and requires ECG evaluation.\n",
                encoding="utf-8",
            )
            chunks = ingest_path(source, source_type="guideline")
        hits = HybridRetriever(chunks).search("heart attack", top_k=3)

        self.assertTrue(hits)
        self.assertIn("myocardial", hits[0].matched_terms)


if __name__ == "__main__":
    unittest.main()

