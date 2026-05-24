from __future__ import annotations

import json
import logging
from pathlib import Path

from medreason_graph.embeddings import (
    DEFAULT_EMBED_DIM,
    EmbeddingConfig,
    TransformerEmbedder,
    chunk_embedding_text,
    resolve_embedding_config,
    text_to_hash_embedding,
)
from medreason_graph.lexicon import SOURCE_TYPE_WEIGHT
from medreason_graph.logging_utils import log_event
from medreason_graph.models import RetrievalHit, SourceChunk
from medreason_graph.query import QueryPart
from medreason_graph.retrieval_scoring import metadata_boost_parts
from medreason_graph.text import detect_concepts, expand_query_terms, tokenize

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1


def build_faiss_index(
    chunks: list[SourceChunk],
    index_path: str | Path,
    *,
    dim: int = DEFAULT_EMBED_DIM,
    embedding_preset: str = "hash",
    query_model: str | None = None,
    document_model: str | None = None,
    pooling: str | None = None,
    query_max_length: int | None = None,
    document_max_length: int | None = None,
    batch_size: int = 8,
) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("FAISS retrieval requires numpy. Install with: pip install -e '.[vector]'") from exc

    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = resolve_embedding_config(
        embedding_preset,
        query_model=query_model,
        document_model=document_model,
        pooling=pooling,
        query_max_length=query_max_length,
        document_max_length=document_max_length,
        dim=dim if embedding_preset == "hash" else None,
    )
    texts = [chunk_embedding_text(chunk) for chunk in chunks]
    if config.backend == "hash":
        vectors = np.vstack([text_to_hash_embedding(text, dim=config.dim or dim) for text in texts]).astype("float32")
    elif config.backend == "transformer":
        assert config.document_model is not None
        embedder = TransformerEmbedder(
            config.document_model,
            pooling=config.document_pooling,
            max_length=config.document_max_length,
        )
        vectors = embedder.encode_texts(texts, batch_size=batch_size).astype("float32")
        config = EmbeddingConfig(**{**config.to_dict(), "dim": int(vectors.shape[1])})
    else:
        raise ValueError(f"unsupported embedding backend: {config.backend}")
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS retrieval requires faiss-cpu. Install with: pip install -e '.[vector]'") from exc

    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    faiss.write_index(index, str(path))
    _chunks_path(path).write_text(json.dumps([chunk.to_dict() for chunk in chunks]), encoding="utf-8")
    _meta_path(path).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "chunk_count": len(chunks),
                "dim": int(vectors.shape[1]),
                "embedding": config.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class FAISSRetriever:
    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self.chunks = [SourceChunk.from_dict(item) for item in json.loads(_chunks_path(self.index_path).read_text(encoding="utf-8"))]
        self.meta = _load_meta(self.index_path)
        self.embedding_config = EmbeddingConfig.from_dict(self.meta.get("embedding", {"backend": "hash", "preset": "hash", "dim": DEFAULT_EMBED_DIM}))
        self._query_embedder = None
        if self.embedding_config.backend == "transformer":
            assert self.embedding_config.query_model is not None
            self._query_embedder = TransformerEmbedder(
                self.embedding_config.query_model,
                pooling=self.embedding_config.query_pooling,
                max_length=self.embedding_config.query_max_length,
            )
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS retrieval requires faiss-cpu. Install with: pip install -e '.[vector]'") from exc

        self.index = faiss.read_index(str(self.index_path))
        self.dim = self.index.d

    def close(self) -> None:
        return None

    def fused_search(
        self,
        query_parts: list[QueryPart],
        *,
        top_k: int = 16,
        source_types: set[str] | None = None,
    ) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        for part in query_parts:
            hits = self._search_part(part, top_k=max(top_k * 4, 30), source_types=source_types)
            for hit in hits:
                existing = merged.get(hit.chunk.id)
                if existing is None:
                    merged[hit.chunk.id] = hit
                    continue
                score_parts = dict(existing.score_parts)
                for key, value in hit.score_parts.items():
                    score_parts[key] = round(score_parts.get(key, 0.0) + value, 6)
                merged[hit.chunk.id] = RetrievalHit(
                    chunk=existing.chunk,
                    score=round(existing.score + hit.score, 6),
                    matched_terms=sorted(set(existing.matched_terms) | set(hit.matched_terms)),
                    score_parts=score_parts,
                    query_labels=sorted(set(existing.query_labels) | set(hit.query_labels)),
                )
        hits = sorted(merged.values(), key=lambda hit: hit.score, reverse=True)
        log_event(
            logger,
            "faiss_retrieval_fused_search",
            query_parts=[part.label for part in query_parts],
            hits=len(hits),
            top_k=top_k,
            index=str(self.index_path),
        )
        return hits[:top_k]

    def _search_part(
        self,
        part: QueryPart,
        *,
        top_k: int,
        source_types: set[str] | None,
    ) -> list[RetrievalHit]:
        query_vector = self._encode_query(part.text)
        distances, indices = self.index.search(query_vector, min(max(top_k * 4, top_k), len(self.chunks)))
        query_terms = sorted(set(expand_query_terms(part.text)))
        hits: list[RetrievalHit] = []
        for distance, index in zip(distances[0].tolist(), indices[0].tolist()):
            if index < 0:
                continue
            chunk = self.chunks[index]
            if chunk.metadata.get("is_noise"):
                continue
            if source_types and chunk.source_type not in source_types:
                continue
            vector_score = max(float(distance), 0.0)
            quality_score = SOURCE_TYPE_WEIGHT.get(chunk.source_type, SOURCE_TYPE_WEIGHT["unknown"])
            section_score = part.section_boosts.get(chunk.section_type, 0.0)
            condition_score = _condition_overlap_score(" ".join([chunk.title, *chunk.section_path, chunk.text]), part.condition_boosts)
            matched_terms = sorted(set(query_terms) & set(tokenize(" ".join([chunk.title, *chunk.section_path, chunk.text]))))
            coverage_score = _coverage_score(query_terms, matched_terms)
            metadata_scores = metadata_boost_parts(part, chunk)
            score = part.weight * (
                (0.58 * vector_score)
                + (0.16 * coverage_score)
                + (0.1 * quality_score)
                + section_score
                + condition_score
                + sum(metadata_scores.values())
            )
            score_parts = {
                f"{part.label}.faiss_ip": round(part.weight * 0.58 * vector_score, 6),
                f"{part.label}.coverage": round(part.weight * 0.16 * coverage_score, 6),
                f"{part.label}.source_quality": round(part.weight * 0.1 * quality_score, 6),
                f"{part.label}.section": round(part.weight * section_score, 6),
                f"{part.label}.condition": round(part.weight * condition_score, 6),
            }
            score_parts.update(
                {
                    f"{part.label}.{name}": round(part.weight * value, 6)
                    for name, value in metadata_scores.items()
                    if value > 0
                }
            )
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=round(score, 6),
                    matched_terms=matched_terms,
                    query_labels=[part.label],
                    score_parts=score_parts,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _encode_query(self, text: str):
        if self.embedding_config.backend in {"hash", "hashing-v1"}:
            return text_to_hash_embedding(text, dim=self.dim).reshape(1, -1).astype("float32")
        if self.embedding_config.backend == "transformer":
            if self._query_embedder is None:
                assert self.embedding_config.query_model is not None
                self._query_embedder = TransformerEmbedder(
                    self.embedding_config.query_model,
                    pooling=self.embedding_config.query_pooling,
                    max_length=self.embedding_config.query_max_length,
                )
            return self._query_embedder.encode_texts([text], batch_size=1).astype("float32")
        raise ValueError(f"unsupported embedding backend: {self.embedding_config.backend}")


def _chunks_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".chunks.json")


def _meta_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".meta.json")


def _load_meta(index_path: Path) -> dict:
    meta_path = _meta_path(index_path)
    if not meta_path.exists():
        return {"embedding": {"backend": "hash", "preset": "hash", "dim": DEFAULT_EMBED_DIM}}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _coverage_score(query_terms: list[str], matched_terms: list[str]) -> float:
    if not query_terms:
        return 0.0
    return min(1.0, len(set(matched_terms)) / len(set(query_terms)))


def _condition_overlap_score(text: str, condition_boosts: set[str]) -> float:
    if not condition_boosts:
        return 0.0
    concepts = detect_concepts(text, kind="condition")
    overlap = concepts & condition_boosts
    if not overlap:
        return 0.0
    return min(0.45, 0.18 * len(overlap))
