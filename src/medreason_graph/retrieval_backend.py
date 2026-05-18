from __future__ import annotations

from pathlib import Path
from typing import Protocol

from medreason_graph.models import RetrievalHit, SourceChunk
from medreason_graph.query import QueryPart
from medreason_graph.faiss_retrieval import FAISSRetriever
from medreason_graph.retrieval import HybridRetriever
from medreason_graph.sqlite_retrieval import SQLiteFTSRetriever


class RetrievalBackend(Protocol):
    chunks: list[SourceChunk]

    def fused_search(
        self,
        query_parts: list[QueryPart],
        *,
        top_k: int = 16,
        source_types: set[str] | None = None,
    ) -> list[RetrievalHit]:
        ...


def create_retriever(kind: str, chunks: list[SourceChunk], index_path: str | Path | None = None) -> RetrievalBackend:
    if kind == "memory":
        return HybridRetriever(chunks)
    if kind == "sqlite":
        if not index_path:
            raise ValueError("index_path is required for sqlite retriever")
        return SQLiteFTSRetriever(index_path)
    if kind == "faiss":
        if not index_path:
            raise ValueError("index_path is required for faiss retriever")
        return FAISSRetriever(index_path)
    raise ValueError(f"unsupported retriever kind: {kind}")


def close_retriever(retriever: RetrievalBackend | None) -> None:
    if retriever is not None and hasattr(retriever, "close"):
        retriever.close()
