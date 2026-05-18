from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from medreason_graph.ingestion import ingest_path


class RealCorpusIngestionTest(unittest.TestCase):
    def test_html_ingestion_preserves_title_section_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "guideline.html"
            path.write_text(
                "<html><head><title>Chest Pain Guideline</title></head>"
                "<body><h1>Chest Pain Guideline</h1><h2>Diagnostic Tests</h2>"
                "<p>Initial evaluation should obtain ECG and troponin for acute coronary syndrome.</p>"
                "</body></html>",
                encoding="utf-8",
            )
            chunks = ingest_path(path, source_type="guideline")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Chest Pain Guideline")
        self.assertEqual(chunks[0].metadata["format"], "html")
        self.assertEqual(chunks[0].section_type, "diagnostic_criteria")
        self.assertIn("Diagnostic Tests", chunks[0].section_path)

    def test_docx_ingestion_extracts_paragraph_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "acs.docx"
            _write_minimal_docx(
                path,
                [
                    "Acute Coronary Syndrome",
                    "Symptoms and Presentation",
                    "Chest pain and diaphoresis support acute coronary syndrome.",
                ],
            )
            chunks = ingest_path(path, source_type="textbook")

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].metadata["format"], "docx")
        self.assertTrue(any("Chest pain" in chunk.text for chunk in chunks))

    def test_directory_ingestion_deduplicates_repeated_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text = "# Duplicate Guideline\n\n## Symptoms\n\nChest pain supports acute coronary syndrome.\n"
            (root / "one.md").write_text(text, encoding="utf-8")
            (root / "two.md").write_text(text, encoding="utf-8")
            chunks = ingest_path(root, source_type="guideline")

        self.assertEqual(len(chunks), 1)


def _write_minimal_docx(path: Path, paragraphs: list[str]) -> None:
    paragraph_xml = "".join(
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>"
        for paragraph in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraph_xml}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        archive.writestr("word/document.xml", document_xml)


if __name__ == "__main__":
    unittest.main()
