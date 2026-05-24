from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from medreason_graph.ingestion import ingest_path
from medreason_graph.storage import save_chunks


ALLOWED_DOMAINS = {
    "www.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "www.nhlbi.nih.gov",
    "nhlbi.nih.gov",
    "www.cdc.gov",
    "cdc.gov",
    "www.fda.gov",
    "fda.gov",
    "www.who.int",
    "who.int",
}
TAG_LIST_FIELDS = ("presentation_tags", "condition_tags", "specialty_tags")


def load_source_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def download_allowlisted_sources(manifest_path: str | Path, out_dir: str | Path, *, delay_seconds: float = 0.5) -> dict[str, Any]:
    manifest = load_source_manifest(manifest_path)
    output_root = Path(out_dir)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []

    for source in manifest.get("sources", []):
        _validate_source(source)
        destination = raw_dir / source["file_name"]
        body, headers = _download(source["url"])
        destination.write_bytes(body)
        record = {
            **source,
            "path": str(destination),
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "content_type": headers.get("Content-Type") or headers.get("content-type"),
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        downloaded.append(record)
        time.sleep(delay_seconds)

    downloaded_manifest = {
        "name": manifest.get("name", "downloaded-open-medical-corpus"),
        "description": manifest.get("description", ""),
        "allowed_domains": sorted(ALLOWED_DOMAINS),
        "sources": downloaded,
    }
    (output_root / "downloaded_manifest.json").write_text(json.dumps(downloaded_manifest, indent=2), encoding="utf-8")
    return downloaded_manifest


def build_downloaded_corpus(downloaded_manifest_path: str | Path, out_path: str | Path) -> int:
    manifest = load_source_manifest(downloaded_manifest_path)
    all_chunks = []
    for source in manifest.get("sources", []):
        chunks = ingest_path(source["path"], source_type=source.get("source_type", "unknown"))
        enriched = []
        for chunk in chunks:
            metadata = {
                **chunk.metadata,
                "source_manifest_id": source["id"],
                "provider": source.get("provider"),
                "license": source.get("license"),
                "url": source.get("url"),
                "source_pack": source.get("source_pack"),
                "presentation_tags": source.get("presentation_tags", []),
                "condition_tags": source.get("condition_tags", []),
                "specialty_tags": source.get("specialty_tags", []),
                "sha256": source.get("sha256"),
            }
            enriched.append(
                type(chunk)(
                    id=chunk.id,
                    source_id=chunk.source_id,
                    title=source.get("title") or chunk.title,
                    source_type=chunk.source_type,
                    section_path=chunk.section_path,
                    section_type=chunk.section_type,
                    paragraph_index=chunk.paragraph_index,
                    text=chunk.text,
                    publication_date=chunk.publication_date,
                    authors=chunk.authors,
                    metadata=metadata,
                )
            )
        all_chunks.extend(enriched)
    save_chunks(all_chunks, out_path)
    return len(all_chunks)


def _validate_source(source: dict[str, Any]) -> None:
    required = {"id", "title", "url", "file_name", "source_type", "license", "provider", "format"}
    missing = required - set(source)
    if missing:
        raise ValueError(f"source {source.get('id', '<unknown>')} missing required fields: {sorted(missing)}")
    parsed = urllib.parse.urlparse(source["url"])
    if parsed.scheme != "https":
        raise ValueError(f"source {source['id']} must use https")
    if parsed.netloc.lower() not in ALLOWED_DOMAINS:
        raise ValueError(f"source {source['id']} uses non-allowlisted domain: {parsed.netloc}")
    if ".." in Path(source["file_name"]).parts:
        raise ValueError(f"source {source['id']} has unsafe file_name")
    _validate_tags(source)


def _validate_tags(source: dict[str, Any]) -> None:
    source_pack = source.get("source_pack")
    if source_pack is not None and (not isinstance(source_pack, str) or not _is_snake_case_tag(source_pack)):
        raise ValueError(f"source {source['id']} has invalid source_pack tag")
    for field in TAG_LIST_FIELDS:
        value = source.get(field, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"source {source['id']} field {field} must be a list of strings")
        if field != "condition_tags":
            invalid = [item for item in value if not _is_snake_case_tag(item)]
            if invalid:
                raise ValueError(f"source {source['id']} field {field} has invalid tags: {invalid}")


def _is_snake_case_tag(value: str) -> bool:
    if not value:
        return False
    return all(char.islower() or char.isdigit() or char == "_" for char in value)


def _download(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MedReasonGraph/0.1 research-prototype contact=local",
            "Accept": "text/html,application/pdf,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read(), dict(response.headers.items())
