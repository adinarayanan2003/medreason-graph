from __future__ import annotations

import unittest
from pathlib import Path

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.config import load_and_apply_config
from medreason_graph.ingestion import ingest_path
from medreason_graph.models import PatientCase
from medreason_graph.terminology import normalize_medical_term
from medreason_graph.text import detect_concepts


ROOT = Path(__file__).resolve().parents[1]


class TerminologyTest(unittest.TestCase):
    def setUp(self) -> None:
        load_and_apply_config(ROOT / "config" / "default_clinical_config.json")

    def test_synonyms_and_contextual_abbreviations_normalize_to_same_concept(self) -> None:
        heart_attack = normalize_medical_term("heart attack")
        mi = normalize_medical_term("MI", context="crushing chest pain with elevated troponin")

        self.assertEqual(heart_attack.canonical, "acute coronary syndrome")
        self.assertEqual(heart_attack.match_method, "synonym")
        self.assertEqual(mi.canonical, "acute coronary syndrome")
        self.assertEqual(mi.match_method, "contextual_abbreviation")

    def test_ambiguous_abbreviation_without_context_is_flagged(self) -> None:
        normalized = normalize_medical_term("PE")

        self.assertTrue(normalized.ambiguous)
        self.assertIsNone(normalized.canonical)
        self.assertIn("pulmonary embolism", normalized.candidates)

    def test_ambiguous_abbreviation_does_not_silently_match_wrong_context(self) -> None:
        exam_context = "PE was normal on exam with clear lungs."
        pulmonary_context = "PE with pleuritic chest pain and sudden dyspnea."

        self.assertNotIn("pulmonary embolism", detect_concepts(exam_context))
        self.assertIn("pulmonary embolism", detect_concepts(pulmonary_context))

    def test_pleuritic_chest_pain_is_a_symptom_not_a_condition_match(self) -> None:
        text = "Pleuritic chest pain and fever can occur with pneumonia."

        self.assertIn("pleuritic pain", detect_concepts(text, kind="symptom"))
        self.assertNotIn("pulmonary embolism", detect_concepts(text, kind="condition"))

    def test_absent_findings_are_not_used_as_supporting_patient_facts(self) -> None:
        chunks = ingest_path(ROOT / "examples" / "corpus", source_type="guideline")
        case = PatientCase.from_dict(
            {
                "case_id": "absent_dyspnea_case",
                "patient": {},
                "chief_complaint": "chest pain",
                "findings": [
                    {"type": "symptom", "name": "dyspnea", "status": "absent"},
                    {"type": "symptom", "name": "diaphoresis", "status": "present"},
                ],
            }
        )

        result = MedReasonAnalyzer(chunks).analyze(case)
        patient_facts = {fact for step in result.reasoning_steps for fact in step.patient_facts}

        self.assertNotIn("dyspnea", patient_facts)


if __name__ == "__main__":
    unittest.main()
