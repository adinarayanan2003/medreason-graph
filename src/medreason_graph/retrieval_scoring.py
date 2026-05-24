from __future__ import annotations

from typing import Any

from medreason_graph.models import SourceChunk
from medreason_graph.query import QueryPart


def metadata_boost_parts(part: QueryPart, chunk: SourceChunk) -> dict[str, float]:
    presentation_overlap = _tag_overlap(part.presentation_boosts, chunk.metadata.get("presentation_tags", []))
    condition_overlap = _tag_overlap(part.condition_tag_boosts, chunk.metadata.get("condition_tags", []))
    source_pack = chunk.metadata.get("source_pack")
    source_pack_match = bool(source_pack and source_pack in part.source_pack_boosts)
    return {
        "presentation_tag": min(0.24, 0.12 * len(presentation_overlap)),
        "condition_tag": min(0.36, 0.18 * len(condition_overlap)),
        "source_pack": 0.12 if source_pack_match else 0.0,
    }


def metadata_boost_score(part: QueryPart, chunk: SourceChunk) -> float:
    return sum(metadata_boost_parts(part, chunk).values())


def _tag_overlap(boosts: set[str], tags: Any) -> set[str]:
    if not boosts or not isinstance(tags, list):
        return set()
    return {tag for tag in tags if isinstance(tag, str) and tag in boosts}
