from __future__ import annotations

import json
from pathlib import Path

from medreason_graph.models import SourceChunk


def save_chunks(chunks: list[SourceChunk], path: str | Path) -> None:
    Path(path).write_text(json.dumps([chunk.to_dict() for chunk in chunks], indent=2), encoding="utf-8")


def load_chunks(path: str | Path) -> list[SourceChunk]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SourceChunk.from_dict(item) for item in data]

