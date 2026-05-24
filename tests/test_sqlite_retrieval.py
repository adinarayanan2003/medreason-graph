from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.ingestion import ingest_path
from medreason_graph.models import PatientCase, SourceChunk
from medreason_graph.query import QueryPart, decompose_case_query
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

    def test_sqlite_retrieval_boosts_matching_source_metadata_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "corpus.sqlite"
            build_sqlite_fts_index(_tagged_chunks(), index_path)
            query = QueryPart(
                label="case_summary",
                text="shared symptom evidence",
                weight=1.0,
                section_boosts={},
                condition_boosts=set(),
                presentation_boosts={"chest_pain"},
                condition_tag_boosts={"acute coronary syndrome"},
                source_pack_boosts={"chest_pain"},
            )
            retriever = SQLiteFTSRetriever(index_path)
            try:
                hits = retriever.fused_search([query], top_k=2)
            finally:
                retriever.close()

        self.assertEqual(hits[0].chunk.id, "target")
        self.assertIn("case_summary.presentation_tag", hits[0].score_parts)
        self.assertIn("case_summary.condition_tag", hits[0].score_parts)
        self.assertIn("case_summary.source_pack", hits[0].score_parts)


def _tagged_chunks() -> list[SourceChunk]:
    return [
        SourceChunk(
            id="generic",
            source_id="generic_source",
            title="Generic Dyspnea Source",
            source_type="textbook",
            section_path=["Generic Dyspnea Source", "Symptoms"],
            section_type="symptoms",
            paragraph_index=1,
            text="Shared symptom evidence.",
            metadata={
                "source_pack": "dyspnea",
                "presentation_tags": ["dyspnea"],
                "condition_tags": ["pneumonia"],
                "specialty_tags": ["pulmonology"],
            },
        ),
        SourceChunk(
            id="target",
            source_id="target_source",
            title="Tagged Chest Pain Source",
            source_type="textbook",
            section_path=["Tagged Chest Pain Source", "Symptoms"],
            section_type="symptoms",
            paragraph_index=1,
            text="Shared symptom evidence.",
            metadata={
                "source_pack": "chest_pain",
                "presentation_tags": ["chest_pain"],
                "condition_tags": ["acute coronary syndrome"],
                "specialty_tags": ["cardiology"],
            },
        ),
    ]


if __name__ == "__main__":
    unittest.main()
