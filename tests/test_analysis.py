from __future__ import annotations

import unittest
from pathlib import Path

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.evidence import extract_evidence_claims
from medreason_graph.ingestion import ingest_path
from medreason_graph.models import PatientCase, ReasoningStep, RetrievalHit, SourceChunk
from medreason_graph.verifier import verify_reasoning


ROOT = Path(__file__).resolve().parents[1]


class AnalysisTest(unittest.TestCase):
    def test_chest_pain_case_produces_auditable_reasoning(self) -> None:
        chunks = ingest_path(ROOT / "examples" / "corpus", source_type="guideline")
        case = PatientCase.from_dict(
            {
                "case_id": "case_test",
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

        self.assertTrue(result.verifier.passed)
        self.assertGreater(len(result.reasoning_steps), 0)
        self.assertTrue(all(step.uses_evidence for step in result.reasoning_steps))
        self.assertEqual(result.differential[0].condition, "acute coronary syndrome")
        self.assertIn("pulmonary embolism", result.verifier.dangerous_misses_checked)
        self.assertIn("aortic dissection", result.verifier.dangerous_misses_checked)
        top = result.differential[0]
        self.assertIn("ecg", top.missing_evidence)
        self.assertIn("troponin", top.missing_evidence)

    def test_verifier_rejects_uncited_reasoning_step(self) -> None:
        chunks = ingest_path(ROOT / "examples" / "corpus", source_type="guideline")
        case = PatientCase.from_dict(
            {
                "case_id": "case_test",
                "patient": {},
                "chief_complaint": "chest pain",
                "findings": [{"type": "symptom", "name": "diaphoresis", "status": "present"}],
            }
        )
        result = MedReasonAnalyzer(chunks).analyze(case)
        bad_step = ReasoningStep(
            id="rs_bad",
            condition="acute coronary syndrome",
            statement="Unsupported statement.",
            uses_evidence=[],
            patient_facts=["diaphoresis"],
        )

        report = verify_reasoning(case, result.evidence_claims, result.reasoning_steps + [bad_step], result.verifier.dangerous_misses_checked)

        self.assertFalse(report.passed)
        self.assertIn("rs_bad", report.unsupported_claims)

    def test_differential_list_sentence_is_not_treated_as_support(self) -> None:
        case = PatientCase.from_dict(
            {
                "case_id": "case_test",
                "patient": {},
                "chief_complaint": "chest pain",
                "findings": [{"type": "symptom", "name": "chest pain", "status": "present"}],
            }
        )
        chunk = SourceChunk(
            id="chunk",
            source_id="source",
            title="Guideline",
            source_type="guideline",
            section_path=["Guideline", "Differential"],
            section_type="differential",
            paragraph_index=1,
            text="The differential diagnosis of chest pain includes acute coronary syndrome, pulmonary embolism, and aortic dissection.",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["chest", "pain"], score_parts={})

        claims = extract_evidence_claims([hit], case)

        self.assertEqual([claim for claim in claims if claim.polarity == "supports"], [])


if __name__ == "__main__":
    unittest.main()
