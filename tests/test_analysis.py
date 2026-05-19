from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.evidence import extract_evidence_claims, validate_evidence_claim
from medreason_graph.ingestion import ingest_path
from medreason_graph.models import PatientCase, ReasoningStep, RetrievalHit, SourceChunk
from medreason_graph.verifier import verify_evidence_claims, verify_reasoning


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

    def test_llm_extractor_accepts_schema_valid_verbatim_span_claim(self) -> None:
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
            text="Diaphoresis supports acute coronary syndrome.",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["diaphoresis"], score_parts={})

        claims = extract_evidence_claims(
            [hit],
            case,
            extractor="llm",
            llm_command=_fake_llm_command(
                {
                    "claims": [
                        {
                            "claim_type": "supports",
                            "condition": "acute coronary syndrome",
                            "finding": "diaphoresis",
                            "polarity": "supports",
                            "strength": "moderate",
                            "exact_quote": "Diaphoresis supports acute coronary syndrome.",
                            "extraction_confidence": 0.91,
                        }
                    ]
                }
            ),
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].extraction_method, "llm_command_v1")
        self.assertEqual(claims[0].extraction_confidence, 0.91)
        self.assertEqual(validate_evidence_claim(claims[0], chunk.text), [])

    def test_llm_extractor_rejects_non_verbatim_span_claim(self) -> None:
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
            section_path=["Guideline", "Symptoms"],
            section_type="symptoms",
            paragraph_index=1,
            text="Chest pain can occur in acute coronary syndrome.",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["chest", "pain"], score_parts={})

        claims = extract_evidence_claims(
            [hit],
            case,
            extractor="llm",
            llm_command=_fake_llm_command(
                {
                    "claims": [
                        {
                            "claim_type": "supports",
                            "condition": "acute coronary syndrome",
                            "finding": "chest pain",
                            "polarity": "supports",
                            "strength": "moderate",
                            "exact_quote": "This quote is not in the source.",
                            "extraction_confidence": 0.99,
                        }
                    ]
                }
            ),
        )

        self.assertEqual(claims, [])

    def test_llm_extractor_rejects_support_claim_not_grounded_in_patient_facts(self) -> None:
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
            section_path=["Guideline", "Symptoms"],
            section_type="symptoms",
            paragraph_index=1,
            text="Pleuritic pain and sudden dyspnea support considering pulmonary embolism.",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["chest", "pain"], score_parts={})

        claims = extract_evidence_claims(
            [hit],
            case,
            extractor="llm",
            llm_command=_fake_llm_command(
                {
                    "claims": [
                        {
                            "claim_type": "supports",
                            "condition": "pulmonary embolism",
                            "finding": "dyspnea",
                            "polarity": "supports",
                            "strength": "moderate",
                            "exact_quote": "Pleuritic pain and sudden dyspnea support considering pulmonary embolism.",
                            "extraction_confidence": 0.9,
                        }
                    ]
                }
            ),
        )

        self.assertEqual(claims, [])

    def test_llm_extractor_rejects_rules_out_when_quote_says_does_not_exclude(self) -> None:
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
            text="A normal initial test does not exclude acute coronary syndrome when symptoms remain concerning.",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["ecg"], score_parts={})

        claims = extract_evidence_claims(
            [hit],
            case,
            extractor="llm",
            llm_command=_fake_llm_command(
                {
                    "claims": [
                        {
                            "claim_type": "rules_out",
                            "condition": "acute coronary syndrome",
                            "finding": "normal initial test",
                            "polarity": "argues_against",
                            "strength": "weak",
                            "exact_quote": "A normal initial test does not exclude acute coronary syndrome when symptoms remain concerning.",
                            "extraction_confidence": 0.9,
                        }
                    ]
                }
            ),
        )

        self.assertEqual(claims, [])

    def test_claim_verifier_rejects_differential_language_as_support(self) -> None:
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
            text="Chest pain from pulmonary embolism needs to be differentiated from acute coronary syndrome.",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["chest", "pain"], score_parts={})
        claims = extract_evidence_claims(
            [hit],
            case,
            extractor="llm",
            llm_command=_fake_llm_command(
                {
                    "claims": [
                        {
                            "claim_type": "supports",
                            "condition": "acute coronary syndrome",
                            "finding": "chest pain",
                            "polarity": "supports",
                            "strength": "moderate",
                            "exact_quote": "Chest pain from pulmonary embolism needs to be differentiated from acute coronary syndrome.",
                            "extraction_confidence": 0.9,
                        }
                    ]
                }
            ),
        )

        verifications = verify_evidence_claims(claims)

        self.assertEqual(len(verifications), 1)
        self.assertFalse(verifications[0].supported)
        self.assertIn("differential_language_not_support", verifications[0].reasons)

    def test_textbook_title_can_supply_condition_context(self) -> None:
        case = PatientCase.from_dict(
            {
                "case_id": "case_test",
                "patient": {},
                "chief_complaint": "dyspnea",
                "findings": [
                    {"type": "symptom", "name": "shortness of breath", "status": "present"},
                    {"type": "symptom", "name": "cough", "status": "present"},
                    {"type": "symptom", "name": "fever", "status": "present"},
                ],
            }
        )
        chunk = SourceChunk(
            id="chunk",
            source_id="source",
            title="Community-Acquired Pneumonia - StatPearls",
            source_type="textbook",
            section_path=["Community-Acquired Pneumonia - StatPearls", "Clinical Features"],
            section_type="symptoms",
            paragraph_index=1,
            text="Typical symptoms include cough, fever, and shortness of breath.",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["cough"], score_parts={})

        claims = extract_evidence_claims([hit], case)
        verifications = verify_evidence_claims(claims)

        self.assertTrue(claims)
        self.assertTrue(all(claim.condition == "pneumonia" for claim in claims))
        self.assertTrue(all(verification.supported for verification in verifications))

    def test_negative_claim_requires_condition_in_cited_sentence(self) -> None:
        case = PatientCase.from_dict(
            {
                "case_id": "case_test",
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
        hit = RetrievalHit(chunk=chunk, score=1.0, matched_terms=["dyspnea"], score_parts={})

        claims = extract_evidence_claims([hit], case)
        verifications = verify_evidence_claims(claims)

        self.assertTrue(claims)
        self.assertFalse(verifications[0].supported)
        self.assertIn("negative_claim_condition_not_in_source_span", verifications[0].reasons)

    def test_analyzer_filters_failed_claims_before_reasoning(self) -> None:
        chunk = SourceChunk(
            id="chunk",
            source_id="source",
            title="Guideline",
            source_type="guideline",
            section_path=["Guideline", "Differential"],
            section_type="differential",
            paragraph_index=1,
            text="Chest pain from pulmonary embolism needs to be differentiated from acute coronary syndrome.",
        )
        case = PatientCase.from_dict(
            {
                "case_id": "case_test",
                "patient": {},
                "chief_complaint": "chest pain",
                "findings": [{"type": "symptom", "name": "chest pain", "status": "present"}],
            }
        )

        result = MedReasonAnalyzer(
            [chunk],
            evidence_extractor="llm",
            llm_command=_fake_llm_command(
                {
                    "claims": [
                        {
                            "claim_type": "supports",
                            "condition": "acute coronary syndrome",
                            "finding": "chest pain",
                            "polarity": "supports",
                            "strength": "moderate",
                            "exact_quote": "Chest pain from pulmonary embolism needs to be differentiated from acute coronary syndrome.",
                            "extraction_confidence": 0.9,
                        }
                    ]
                }
            ),
        ).analyze(case)

        self.assertEqual(result.evidence_claims, [])
        self.assertTrue(result.claim_verifications)
        self.assertFalse(result.claim_verifications[0].supported)

def _fake_llm_command(response: dict) -> str:
    temp = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False)
    script = Path(temp.name)
    temp.write("import json, sys\n")
    temp.write("_ = sys.stdin.read()\n")
    temp.write(f"print(json.dumps({json.dumps(response)}))\n")
    temp.close()
    return f"{sys.executable} {script}"


if __name__ == "__main__":
    unittest.main()
