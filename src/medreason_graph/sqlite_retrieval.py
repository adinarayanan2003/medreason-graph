from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from medreason_graph.lexicon import SOURCE_TYPE_WEIGHT
from medreason_graph.logging_utils import log_event
from medreason_graph.models import RetrievalHit, SourceChunk
from medreason_graph.query import QueryPart
from medreason_graph.retrieval_scoring import metadata_boost_parts
from medreason_graph.text import detect_concepts, expand_query_terms, term_frequency, tokenize

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1


def build_sqlite_fts_index(chunks: list[SourceChunk], index_path: str | Path) -> None:
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("DROP TABLE IF EXISTS corpus_meta")
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.execute("CREATE TABLE corpus_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "id UNINDEXED, title, section_path, section_type, source_type, text, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        conn.executemany(
            "INSERT INTO chunks (id, payload) VALUES (?, ?)",
            [(chunk.id, json.dumps(chunk.to_dict())) for chunk in chunks],
        )
        conn.executemany(
            "INSERT INTO chunks_fts (id, title, section_path, section_type, source_type, text) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    chunk.id,
                    chunk.title,
                    " ".join(chunk.section_path),
                    chunk.section_type,
                    chunk.source_type,
                    chunk.text,
                )
                for chunk in chunks
            ],
        )
        conn.execute("INSERT INTO corpus_meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        conn.execute("INSERT INTO corpus_meta (key, value) VALUES ('chunk_count', ?)", (str(len(chunks)),))


class SQLiteFTSRetriever:
    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self.conn = sqlite3.connect(self.index_path)
        self.conn.row_factory = sqlite3.Row
        self.chunks = self._load_chunks()

    def close(self) -> None:
        self.conn.close()

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
            "sqlite_retrieval_fused_search",
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
        query_terms = sorted(set(expand_query_terms(part.text)))
        match_query = _fts_query(query_terms)
        if not match_query:
            return []
        rows = self.conn.execute(
            """
            SELECT c.payload, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.id = c.id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match_query, top_k * 4),
        ).fetchall()

        hits: list[RetrievalHit] = []
        for row in rows:
            chunk = SourceChunk.from_dict(json.loads(row["payload"]))
            if chunk.metadata.get("is_noise"):
                continue
            if source_types and chunk.source_type not in source_types:
                continue
            lexical_score = _rank_to_score(float(row["rank"]))
            quality_score = SOURCE_TYPE_WEIGHT.get(chunk.source_type, SOURCE_TYPE_WEIGHT["unknown"])
            section_score = part.section_boosts.get(chunk.section_type, 0.0)
            condition_score = _condition_overlap_score(" ".join([chunk.title, *chunk.section_path, chunk.text]), part.condition_boosts)
            matched_terms = sorted(set(query_terms) & set(tokenize(" ".join([chunk.title, *chunk.section_path, chunk.text]))))
            coverage_score = _coverage_score(query_terms, matched_terms)
            metadata_scores = metadata_boost_parts(part, chunk)
            score = part.weight * (
                (0.62 * lexical_score)
                + (0.16 * coverage_score)
                + (0.1 * quality_score)
                + section_score
                + condition_score
                + sum(metadata_scores.values())
            )
            score_parts = {
                f"{part.label}.fts_bm25": round(part.weight * 0.62 * lexical_score, 6),
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

    def _load_chunks(self) -> list[SourceChunk]:
        rows = self.conn.execute("SELECT payload FROM chunks").fetchall()
        return [SourceChunk.from_dict(json.loads(row["payload"])) for row in rows]


def _fts_query(tokens: list[str]) -> str:
    clean = [token for token in tokens if token]
    if not clean:
        return ""
    return " OR ".join(f'"{token.replace(chr(34), chr(34) + chr(34))}"' for token in clean[:80])


def _rank_to_score(rank: float) -> float:
    # SQLite FTS5 bm25() returns lower values for better matches, commonly negative.
    return 1.0 / (1.0 + max(rank, 0.0) + abs(min(rank, 0.0)))


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
