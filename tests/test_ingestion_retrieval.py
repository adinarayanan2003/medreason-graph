from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medreason_graph.ingestion import ingest_path
from medreason_graph.models import SourceChunk
from medreason_graph.query import QueryPart
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

    def test_hybrid_retrieval_boosts_matching_source_metadata_tags(self) -> None:
        chunks = _tagged_chunks()
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

        hits = HybridRetriever(chunks).fused_search([query], top_k=2)

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
