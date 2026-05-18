from __future__ import annotations

import math
import logging
from collections import Counter

from medreason_graph.lexicon import SOURCE_TYPE_WEIGHT
from medreason_graph.logging_utils import log_event
from medreason_graph.models import RetrievalHit, SourceChunk
from medreason_graph.query import QueryPart
from medreason_graph.text import detect_concepts
from medreason_graph.text import cosine_from_counts, expand_query_terms, term_frequency, tokenize

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, chunks: list[SourceChunk]):
        self.chunks = chunks
        self.chunk_tokens = {chunk.id: tokenize(chunk.text) for chunk in chunks}
        self.chunk_counts = {chunk_id: term_frequency(tokens) for chunk_id, tokens in self.chunk_tokens.items()}
        self.doc_freq = self._doc_frequency()
        self.avg_doc_len = (
            sum(len(tokens) for tokens in self.chunk_tokens.values()) / len(self.chunk_tokens)
            if self.chunk_tokens
            else 0.0
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: set[str] | None = None,
        section_types: set[str] | None = None,
    ) -> list[RetrievalHit]:
        query_tokens = expand_query_terms(query)
        query_counts = term_frequency(query_tokens)
        hits: list[RetrievalHit] = []
        for chunk in self.chunks:
            if source_types and chunk.source_type not in source_types:
                continue
            if section_types and chunk.section_type not in section_types:
                continue
            bm25_score = self._bm25(query_tokens, chunk)
            semantic_score = cosine_from_counts(query_counts, self.chunk_counts.get(chunk.id, {}))
            quality_score = SOURCE_TYPE_WEIGHT.get(chunk.source_type, SOURCE_TYPE_WEIGHT["unknown"])
            score = (0.65 * bm25_score) + (0.25 * semantic_score) + (0.1 * quality_score)
            if score <= 0:
                continue
            matched = sorted(set(query_tokens) & set(self.chunk_tokens.get(chunk.id, [])))
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=round(score, 6),
                    matched_terms=matched,
                    score_parts={
                        "bm25": round(bm25_score, 6),
                        "semantic": round(semantic_score, 6),
                        "source_quality": round(quality_score, 6),
                    },
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        log_event(
            logger,
            "retrieval_search",
            query=query,
            candidate_chunks=len(self.chunks),
            hits=len(hits),
            top_k=top_k,
        )
        return hits[:top_k]

    def fused_search(
        self,
        query_parts: list[QueryPart],
        *,
        top_k: int = 16,
        source_types: set[str] | None = None,
    ) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        for part in query_parts:
            hits = self._search_part(part, top_k=max(top_k * 3, 20), source_types=source_types)
            for hit in hits:
                existing = merged.get(hit.chunk.id)
                if existing is None:
                    merged[hit.chunk.id] = hit
                    continue
                score = existing.score + hit.score
                matched_terms = sorted(set(existing.matched_terms) | set(hit.matched_terms))
                score_parts = dict(existing.score_parts)
                for key, value in hit.score_parts.items():
                    score_parts[key] = round(score_parts.get(key, 0.0) + value, 6)
                merged[hit.chunk.id] = RetrievalHit(
                    chunk=existing.chunk,
                    score=round(score, 6),
                    matched_terms=matched_terms,
                    score_parts=score_parts,
                    query_labels=sorted(set(existing.query_labels) | set(hit.query_labels)),
                )
        hits = sorted(merged.values(), key=lambda hit: hit.score, reverse=True)
        log_event(
            logger,
            "retrieval_fused_search",
            query_parts=[part.label for part in query_parts],
            candidate_chunks=len(self.chunks),
            hits=len(hits),
            top_k=top_k,
        )
        return hits[:top_k]

    def _search_part(
        self,
        part: QueryPart,
        *,
        top_k: int,
        source_types: set[str] | None,
    ) -> list[RetrievalHit]:
        query_tokens = expand_query_terms(part.text)
        query_counts = term_frequency(query_tokens)
        hits: list[RetrievalHit] = []
        for chunk in self.chunks:
            if chunk.metadata.get("is_noise"):
                continue
            if source_types and chunk.source_type not in source_types:
                continue
            bm25_score = self._bm25(query_tokens, chunk)
            semantic_score = cosine_from_counts(query_counts, self.chunk_counts.get(chunk.id, {}))
            quality_score = SOURCE_TYPE_WEIGHT.get(chunk.source_type, SOURCE_TYPE_WEIGHT["unknown"])
            section_score = part.section_boosts.get(chunk.section_type, 0.0)
            condition_score = _condition_overlap_score(" ".join([chunk.title, *chunk.section_path, chunk.text]), part.condition_boosts)
            score = part.weight * (
                (0.58 * bm25_score)
                + (0.18 * semantic_score)
                + (0.1 * quality_score)
                + section_score
                + condition_score
            )
            if score <= 0:
                continue
            matched = sorted(set(query_tokens) & set(self.chunk_tokens.get(chunk.id, [])))
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=round(score, 6),
                    matched_terms=matched,
                    query_labels=[part.label],
                    score_parts={
                        f"{part.label}.bm25": round(part.weight * 0.58 * bm25_score, 6),
                        f"{part.label}.semantic": round(part.weight * 0.18 * semantic_score, 6),
                        f"{part.label}.source_quality": round(part.weight * 0.1 * quality_score, 6),
                        f"{part.label}.section": round(part.weight * section_score, 6),
                        f"{part.label}.condition": round(part.weight * condition_score, 6),
                    },
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _doc_frequency(self) -> dict[str, int]:
        doc_freq: Counter[str] = Counter()
        for tokens in self.chunk_tokens.values():
            doc_freq.update(set(tokens))
        return dict(doc_freq)

    def _bm25(self, query_tokens: list[str], chunk: SourceChunk) -> float:
        tokens = self.chunk_tokens.get(chunk.id, [])
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        k1 = 1.5
        b = 0.75
        score = 0.0
        doc_len = len(tokens)
        for token in set(query_tokens):
            tf = counts.get(token, 0)
            if not tf:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + ((len(self.chunks) - df + 0.5) / (df + 0.5)))
            denom = tf + k1 * (1 - b + b * (doc_len / (self.avg_doc_len or 1)))
            score += idf * ((tf * (k1 + 1)) / denom)
        return score


def _condition_overlap_score(text: str, condition_boosts: set[str]) -> float:
    if not condition_boosts:
        return 0.0
    concepts = detect_concepts(text, kind="condition")
    overlap = concepts & condition_boosts
    if not overlap:
        return 0.0
    return min(0.45, 0.18 * len(overlap))
