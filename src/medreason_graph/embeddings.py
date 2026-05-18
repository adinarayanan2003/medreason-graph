from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

from medreason_graph.text import expand_query_terms, tokenize


DEFAULT_EMBED_DIM = 384


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str
    preset: str
    query_model: str | None
    document_model: str | None
    query_pooling: str = "cls"
    document_pooling: str = "cls"
    query_max_length: int = 64
    document_max_length: int = 512
    dim: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "preset": self.preset,
            "query_model": self.query_model,
            "document_model": self.document_model,
            "query_pooling": self.query_pooling,
            "document_pooling": self.document_pooling,
            "query_max_length": self.query_max_length,
            "document_max_length": self.document_max_length,
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> "EmbeddingConfig":
        if isinstance(data, str):
            if data in {"hash", "hashing-v1"}:
                data = {"backend": data, "preset": "hash", "dim": DEFAULT_EMBED_DIM}
            else:
                raise ValueError(f"unsupported embedding metadata: {data}")
        return cls(
            backend=data.get("backend", data.get("embedding", "hashing-v1")),
            preset=data.get("preset", "hash"),
            query_model=data.get("query_model"),
            document_model=data.get("document_model"),
            query_pooling=data.get("query_pooling", "cls"),
            document_pooling=data.get("document_pooling", "cls"),
            query_max_length=int(data.get("query_max_length", 64)),
            document_max_length=int(data.get("document_max_length", 512)),
            dim=data.get("dim"),
        )


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "hash": {
        "backend": "hash",
        "query_model": None,
        "document_model": None,
        "query_pooling": "hash",
        "document_pooling": "hash",
        "query_max_length": 0,
        "document_max_length": 0,
        "dim": DEFAULT_EMBED_DIM,
    },
    "medcpt": {
        "backend": "transformer",
        "query_model": "ncbi/MedCPT-Query-Encoder",
        "document_model": "ncbi/MedCPT-Article-Encoder",
        "query_pooling": "cls",
        "document_pooling": "cls",
        "query_max_length": 64,
        "document_max_length": 512,
    },
    "sapbert": {
        "backend": "transformer",
        "query_model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "document_model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "query_pooling": "cls",
        "document_pooling": "cls",
        "query_max_length": 128,
        "document_max_length": 256,
    },
    "bioclinicalbert": {
        "backend": "transformer",
        "query_model": "emilyalsentzer/Bio_ClinicalBERT",
        "document_model": "emilyalsentzer/Bio_ClinicalBERT",
        "query_pooling": "cls",
        "document_pooling": "cls",
        "query_max_length": 128,
        "document_max_length": 512,
    },
}


def resolve_embedding_config(
    preset: str = "hash",
    *,
    query_model: str | None = None,
    document_model: str | None = None,
    pooling: str | None = None,
    query_max_length: int | None = None,
    document_max_length: int | None = None,
    dim: int | None = None,
) -> EmbeddingConfig:
    if preset not in MODEL_PRESETS:
        raise ValueError(f"unknown embedding preset: {preset}")
    base = dict(MODEL_PRESETS[preset])
    if query_model:
        base["query_model"] = query_model
    if document_model:
        base["document_model"] = document_model
    if pooling:
        base["query_pooling"] = pooling
        base["document_pooling"] = pooling
    if query_max_length:
        base["query_max_length"] = query_max_length
    if document_max_length:
        base["document_max_length"] = document_max_length
    if dim:
        base["dim"] = dim
    return EmbeddingConfig(preset=preset, **base)


def text_to_hash_embedding(text: str, *, dim: int = DEFAULT_EMBED_DIM, expand_terms: bool = True):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Vector embeddings require numpy. Install with: pip install -e '.[vector]'") from exc

    tokens = expand_query_terms(text) if expand_terms else tokenize(text)
    vector = np.zeros(dim, dtype="float32")
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        index = value % dim
        sign = 1.0 if ((value >> 8) & 1) == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(float((vector * vector).sum()))
    if norm:
        vector /= norm
    return vector


def chunk_embedding_text(chunk) -> str:
    return " ".join([chunk.title, *chunk.section_path, chunk.section_type, chunk.source_type, chunk.text])


class TransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        pooling: str = "cls",
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        # macOS wheels for FAISS and PyTorch can load separate OpenMP runtimes in
        # the same process. This keeps the local FAISS+transformer prototype
        # usable; production deployments should prefer a clean Linux container.
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformer medical encoders require torch and transformers. Install with: pip install -e '.[medical-encoders]'"
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_texts(self, texts: list[str], *, batch_size: int = 8):
        import numpy as np

        vectors = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded).last_hidden_state
                if self.pooling == "mean":
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                elif self.pooling == "cls":
                    pooled = output[:, 0, :]
                else:
                    raise ValueError(f"unsupported pooling: {self.pooling}")
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.detach().cpu().numpy().astype("float32"))
        if not vectors:
            return np.zeros((0, 0), dtype="float32")
        return np.vstack(vectors).astype("float32")
