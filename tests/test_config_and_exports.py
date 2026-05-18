from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from medreason_graph.analyzer import MedReasonAnalyzer
from medreason_graph.config import load_and_apply_config
from medreason_graph.graph import export_cytoscape, export_graphviz_dot
from medreason_graph.ingestion import ingest_path
from medreason_graph.models import PatientCase


ROOT = Path(__file__).resolve().parents[1]


def _sample_result():
    load_and_apply_config(ROOT / "config" / "default_clinical_config.json")
    chunks = ingest_path(ROOT / "examples" / "corpus", source_type="guideline")
    case = PatientCase.from_dict(json.loads((ROOT / "examples" / "cases" / "chest_pain.json").read_text(encoding="utf-8")))
    return MedReasonAnalyzer(chunks).analyze(case)


class ConfigAndExportsTest(unittest.TestCase):
    def test_default_config_preserves_golden_summary(self) -> None:
        result = _sample_result()
        golden = json.loads((ROOT / "examples" / "golden" / "chest_pain_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(result.case_id, golden["case_id"])
        self.assertEqual(result.verifier.passed, golden["verifier_passed"])
        self.assertEqual([item.condition for item in result.differential[:3]], golden["top_differential"])
        self.assertEqual(result.verifier.dangerous_misses_checked, golden["dangerous_misses_checked"])
        self.assertEqual(result.differential[0].missing_evidence, golden["top_missing_evidence"])
        self.assertGreaterEqual(len(result.reasoning_steps), golden["minimum_reasoning_steps"])
        self.assertGreaterEqual(len(result.evidence_claims), golden["minimum_evidence_claims"])

    def test_custom_config_can_add_synonym_without_python_changes(self) -> None:
        default_config = json.loads((ROOT / "config" / "default_clinical_config.json").read_text(encoding="utf-8"))
        default_config["concepts"]["acute coronary syndrome"]["synonyms"].append("blocked heart artery syndrome")
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "clinical_config.json"
            config_path.write_text(json.dumps(default_config), encoding="utf-8")
            load_and_apply_config(config_path)
            chunks = ingest_path(ROOT / "examples" / "corpus", source_type="guideline")
            case = PatientCase.from_dict(
                {
                    "case_id": "custom_synonym_case",
                    "patient": {},
                    "chief_complaint": "blocked heart artery syndrome",
                    "findings": [{"type": "symptom", "name": "diaphoresis", "status": "present"}],
                }
            )

        result = MedReasonAnalyzer(chunks).analyze(case)

        self.assertEqual(result.differential[0].condition, "acute coronary syndrome")

    def test_graph_exports_have_viewer_shapes(self) -> None:
        result = _sample_result()
        cytoscape = export_cytoscape(result.graph)
        dot = export_graphviz_dot(result.graph)

        self.assertIn("elements", cytoscape)
        self.assertTrue(any("source" in element["data"] for element in cytoscape["elements"]))
        self.assertTrue(dot.startswith("digraph MedReasonGraph"))
        self.assertIn("uses_evidence", dot)


if __name__ == "__main__":
    unittest.main()

