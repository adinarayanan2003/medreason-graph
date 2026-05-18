from __future__ import annotations

import unittest
from pathlib import Path

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.evidence import extract_evidence_claims, validate_evidence_claim
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

    def test_differentiated_from_sentence_is_not_treated_as_support(self) -> None:
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
            text="Chest pain from pulmonary embolism needs to be differentiated from acute coronary syndrome and aortic dissection.",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["chest", "pain"], score_parts={})

        claims = extract_evidence_claims([hit], case)

        self.assertEqual([claim for claim in claims if claim.polarity == "supports"], [])

    def test_extracted_claims_have_valid_schema_and_source_span(self) -> None:
        case = PatientCase.from_dict(
            {
                "case_id": "case_test",
                "patient": {},
                "chief_complaint": "crushing chest pain",
                "findings": [{"type": "symptom", "name": "diaphoresis", "status": "present"}],
            }
        )
        chunk = SourceChunk(
            id="chunk",
            source_id="source",
            title="Guideline",
            source_type="guideline",
            section_path=["Guideline", "Symptoms"],
            section_type="symptoms",
            paragraph_index=1,
            text="Acute coronary syndrome often presents with chest pain. Diaphoresis supports acute coronary syndrome.",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["diaphoresis"], score_parts={})

        claims = extract_evidence_claims([hit], case)

        self.assertTrue(claims)
        for claim in claims:
            self.assertEqual(validate_evidence_claim(claim, chunk.text), [])
            self.assertEqual(chunk.text[claim.source_span_start:claim.source_span_end], claim.sentence)
            self.assertGreater(claim.extraction_confidence, 0)
            self.assertEqual(claim.extraction_method, "deterministic_cue_v1")

    def test_rule_out_sentence_is_structured_as_rules_out(self) -> None:
        case = PatientCase.from_dict(
            {
                "case_id": "case_test",
                "patient": {},
                "chief_complaint": "chest pain",
                "findings": [{"type": "test", "name": "ECG", "status": "present"}],
            }
        )
        chunk = SourceChunk(
            id="chunk",
            source_id="source",
            title="Guideline",
            source_type="guideline",
            section_path=["Guideline", "Tests"],
            section_type="tests",
            paragraph_index=1,
            text="A diagnostic ECG can rule out acute coronary syndrome when symptoms are low risk.",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["ecg"], score_parts={})

        claims = extract_evidence_claims([hit], case)

        self.assertTrue(any(claim.claim_type == "rules_out" for claim in claims))
        self.assertTrue(all(claim.polarity == "argues_against" for claim in claims))

    def test_does_not_exclude_sentence_is_not_negative_evidence(self) -> None:
        case = PatientCase.from_dict(
            {
                "case_id": "case_test",
                "patient": {},
                "chief_complaint": "chest pain",
                "findings": [{"type": "test", "name": "ECG", "status": "present"}],
            }
        )
        chunk = SourceChunk(
            id="chunk",
            source_id="source",
            title="Guideline",
            source_type="guideline",
            section_path=["Guideline", "Tests"],
            section_type="tests",
            paragraph_index=1,
            text="A normal ECG does not exclude acute coronary syndrome when symptoms remain concerning.",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["ecg"], score_parts={})

        claims = extract_evidence_claims([hit], case)

        self.assertTrue(claims)
        self.assertFalse(any(claim.claim_type == "rules_out" for claim in claims))
        self.assertFalse(any(claim.polarity == "argues_against" for claim in claims))


if __name__ == "__main__":
    unittest.main()
