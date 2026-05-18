from __future__ import annotations

import hashlib
import html
from html.parser import HTMLParser
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from medreason_graph.logging_utils import log_event
from medreason_graph.models import SourceChunk
from medreason_graph.noise import is_noise_chunk
from medreason_graph.text import detect_section_type

logger = logging.getLogger(__name__)


def ingest_path(path: str | Path, source_type: str = "unknown") -> list[SourceChunk]:
    target = Path(path)
    if target.is_dir():
        chunks: list[SourceChunk] = []
        seen_texts: set[str] = set()
        for child in sorted(target.rglob("*")):
            if child.suffix.lower() in {".md", ".txt", ".json", ".html", ".htm", ".docx", ".pdf"}:
                for chunk in ingest_path(child, source_type=source_type):
                    text_key = _content_key(chunk.text)
                    if text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)
                    chunks.append(chunk)
        log_event(logger, "ingest_directory", path=str(target), chunks=len(chunks), source_type=source_type)
        return chunks
    if target.suffix.lower() == ".json":
        chunks = _ingest_json(target, default_source_type=source_type)
    elif target.suffix.lower() in {".html", ".htm"}:
        chunks = _ingest_html(target, source_type=source_type)
    elif target.suffix.lower() == ".docx":
        chunks = _ingest_docx(target, source_type=source_type)
    elif target.suffix.lower() == ".pdf":
        chunks = _ingest_pdf(target, source_type=source_type)
    else:
        chunks = _ingest_text(target, source_type=source_type)
    log_event(logger, "ingest_file", path=str(target), chunks=len(chunks), source_type=source_type)
    return chunks


def _ingest_json(path: Path, default_source_type: str) -> list[SourceChunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        chunks = []
        for item in data:
            if {"id", "source_id", "title", "source_type", "section_path", "section_type", "paragraph_index", "text"} <= set(item):
                chunks.append(SourceChunk.from_dict(item))
            else:
                chunks.extend(_chunks_from_record(item, path, default_source_type))
        return chunks
    return _chunks_from_record(data, path, default_source_type)


def _chunks_from_record(record: dict[str, Any], path: Path, default_source_type: str) -> list[SourceChunk]:
    title = record.get("title") or path.stem.replace("_", " ").title()
    source_type = record.get("source_type", default_source_type)
    source_id = record.get("source_id") or _stable_id(str(path), title)
    text = record.get("text", "")
    publication_date = record.get("publication_date")
    authors = record.get("authors", [])
    metadata = {key: value for key, value in record.items() if key not in {"text", "title", "source_type", "source_id"}}
    return _chunk_text(
        text,
        source_id=source_id,
        title=title,
        source_type=source_type,
        publication_date=publication_date,
        authors=authors,
        metadata=metadata,
    )


def _ingest_text(path: Path, source_type: str) -> list[SourceChunk]:
    text = path.read_text(encoding="utf-8")
    title = _title_from_text(text) or path.stem.replace("_", " ").title()
    source_id = _stable_id(str(path), title)
    return _chunk_text(text, source_id=source_id, title=title, source_type=source_type, metadata={"path": str(path)})


def _ingest_html(path: Path, source_type: str) -> list[SourceChunk]:
    parser = _ReadableHTMLParser()
    parser.feed(path.read_text(encoding="utf-8"))
    text = parser.to_markdownish_text()
    title = parser.title or _title_from_text(text) or path.stem.replace("_", " ").title()
    source_id = _stable_id(str(path), title)
    return _chunk_text(text, source_id=source_id, title=title, source_type=source_type, metadata={"path": str(path), "format": "html"})


def _ingest_docx(path: Path, source_type: str) -> list[SourceChunk]:
    text = _extract_docx_text(path)
    title = _title_from_text(text) or path.stem.replace("_", " ").title()
    source_id = _stable_id(str(path), title)
    return _chunk_text(text, source_id=source_id, title=title, source_type=source_type, metadata={"path": str(path), "format": "docx"})


def _ingest_pdf(path: Path, source_type: str) -> list[SourceChunk]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf. Install with: pip install -e '.[corpus]'") from exc

    reader = PdfReader(str(path))
    chunks: list[SourceChunk] = []
    title = path.stem.replace("_", " ").title()
    source_id = _stable_id(str(path), title)
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue
        chunks.extend(
            _chunk_text(
                page_text,
                source_id=source_id,
                title=title,
                source_type=source_type,
                metadata={"path": str(path), "format": "pdf", "page": page_index},
                paragraph_offset=len(chunks),
            )
        )
    return chunks


def _chunk_text(
    text: str,
    *,
    source_id: str,
    title: str,
    source_type: str,
    publication_date: str | None = None,
    authors: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    paragraph_offset: int = 0,
) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    section_path: list[str] = [title]
    paragraph_index = paragraph_offset
    buffer: list[str] = []

    def flush() -> None:
        nonlocal paragraph_index, buffer
        paragraph = "\n".join(buffer).strip()
        if not paragraph:
            buffer = []
            return
        for split_text in _split_long_paragraph(paragraph):
            paragraph_index += 1
            chunk_id = _stable_id(source_id, "|".join(section_path), str(paragraph_index), split_text[:80])
            chunk_metadata = dict(metadata or {})
            chunk_metadata["is_noise"] = is_noise_chunk(split_text, section_path)
            chunks.append(
                SourceChunk(
                    id=chunk_id,
                    source_id=source_id,
                    title=title,
                    source_type=source_type,
                    section_path=list(section_path),
                    section_type=detect_section_type(section_path),
                    paragraph_index=paragraph_index,
                    text=split_text,
                    publication_date=publication_date,
                    authors=authors or [],
                    metadata=chunk_metadata,
                )
            )
        buffer = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if _is_heading(line):
            flush()
            heading = _clean_heading(line)
            level = _heading_level(line)
            if level == 1:
                section_path = [heading]
            else:
                section_path = section_path[: level - 1]
                section_path.append(heading)
            continue
        if not line:
            flush()
            continue
        buffer.append(line)
    flush()
    return chunks


def _split_long_paragraph(text: str, max_chars: int = 1200) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = text.replace(". ", ".\n").splitlines()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        if current and length + len(sentence) > max_chars:
            chunks.append(" ".join(current).strip())
            current = []
            length = 0
        current.append(sentence)
        length += len(sentence)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _title_from_text(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return _clean_heading(stripped)
    return None


def _is_heading(line: str) -> bool:
    if not line:
        return False
    if line.startswith("#"):
        return True
    return len(line) <= 80 and line.endswith(":") and line[:-1].strip().istitle()


def _clean_heading(line: str) -> str:
    return line.strip("#: ").strip()


def _heading_level(line: str) -> int:
    if line.startswith("#"):
        return min(len(line) - len(line.lstrip("#")), 6)
    return 2


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"id_{digest}"


def _content_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _extract_docx_text(path: Path) -> str:
    paragraphs: list[str] = []
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        joined = "".join(texts).strip()
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title: str | None = None
        self._in_title = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            self._parts.append("\n" + ("#" * level) + " ")
        elif tag in {"p", "li", "section", "article", "br"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        value = html.unescape(data).strip()
        if not value:
            return
        if self._in_title:
            self.title = value
        else:
            self._parts.append(value + " ")

    def to_markdownish_text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()
