from __future__ import annotations

import unittest

from medreason_graph.embeddings import DEFAULT_EMBED_DIM, EmbeddingConfig, resolve_embedding_config


class EmbeddingPresetTest(unittest.TestCase):
    def test_medcpt_is_retrieval_default_preset(self) -> None:
        config = resolve_embedding_config("medcpt")

        self.assertEqual(config.backend, "transformer")
        self.assertEqual(config.query_model, "ncbi/MedCPT-Query-Encoder")
        self.assertEqual(config.document_model, "ncbi/MedCPT-Article-Encoder")
        self.assertEqual(config.query_pooling, "cls")

    def test_sapbert_and_bioclinicalbert_presets_are_available(self) -> None:
        sapbert = resolve_embedding_config("sapbert")
        bioclinicalbert = resolve_embedding_config("bioclinicalbert")

        self.assertIn("SapBERT", sapbert.query_model)
        self.assertEqual(bioclinicalbert.query_model, "emilyalsentzer/Bio_ClinicalBERT")

    def test_custom_transformer_model_can_override_preset(self) -> None:
        config = resolve_embedding_config(
            "medcpt",
            query_model="custom/query",
            document_model="custom/document",
            pooling="mean",
            query_max_length=32,
            document_max_length=128,
        )

        self.assertEqual(config.query_model, "custom/query")
        self.assertEqual(config.document_model, "custom/document")
        self.assertEqual(config.query_pooling, "mean")
        self.assertEqual(config.document_max_length, 128)

    def test_legacy_hashing_embedding_metadata_still_loads(self) -> None:
        config = EmbeddingConfig.from_dict("hashing-v1")

        self.assertEqual(config.backend, "hashing-v1")
        self.assertEqual(config.preset, "hash")
        self.assertEqual(config.dim, DEFAULT_EMBED_DIM)


if __name__ == "__main__":
    unittest.main()
