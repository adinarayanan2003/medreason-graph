from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from medreason_graph.source_downloader import build_downloaded_corpus, download_allowlisted_sources
from medreason_graph.storage import load_chunks


class SourceDownloaderTest(unittest.TestCase):
    def test_downloader_rejects_non_allowlisted_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "id": "bad",
                                "title": "Bad",
                                "url": "https://example.com/bad.pdf",
                                "file_name": "bad.pdf",
                                "source_type": "guideline",
                                "license": "unknown",
                                "provider": "unknown",
                                "format": "pdf",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                download_allowlisted_sources(manifest_path, Path(temp_dir) / "out", delay_seconds=0)

    def test_build_downloaded_corpus_enriches_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            source_path = raw / "source.md"
            source_path.write_text(
                "# Open Source\n\n## Symptoms\n\nChest pain supports acute coronary syndrome.\n",
                encoding="utf-8",
            )
            manifest_path = root / "downloaded_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "id": "source",
                                "title": "Open Source",
                                "url": "https://www.ncbi.nlm.nih.gov/books/example",
                                "file_name": "source.md",
                                "path": str(source_path),
                                "source_type": "textbook",
                                "license": "test license",
                                "provider": "NCBI Bookshelf",
                                "format": "markdown",
                                "sha256": "abc",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out = root / "corpus.json"
            count = build_downloaded_corpus(manifest_path, out)
            chunks = load_chunks(out)

        self.assertEqual(count, 1)
        self.assertEqual(chunks[0].title, "Open Source")
        self.assertEqual(chunks[0].metadata["license"], "test license")
        self.assertEqual(chunks[0].metadata["provider"], "NCBI Bookshelf")
        self.assertEqual(chunks[0].metadata["source_manifest_id"], "source")


if __name__ == "__main__":
    unittest.main()

