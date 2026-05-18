#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-4.1-mini"


def main() -> int:
    _load_dotenv(Path.cwd() / ".env")
    try:
        from openai import OpenAI
    except ImportError:
        print(
            json.dumps(
                {
                    "claims": [],
                    "error": "Install OpenAI support with: pip install -e '.[llm-openai]'",
                }
            ),
            file=sys.stderr,
        )
        return 2

    payload = json.loads(sys.stdin.read())
    source = payload.get("source", {})
    text = source.get("text", "")
    if not text:
        print(json.dumps({"claims": []}))
        return 0

    client = OpenAI()
    response = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        input=[
            {
                "role": "system",
                "content": (
                    "You extract medical evidence claims from retrieved reference passages. "
                    "Return only claims directly supported by the source text. Do not diagnose. "
                    "Do not use outside medical knowledge. exact_quote must be copied verbatim "
                    "from source.text."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(_compact_payload(payload), ensure_ascii=False),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "medical_evidence_claims",
                "schema": _response_schema(),
                "strict": True,
            }
        },
    )
    output_text = getattr(response, "output_text", "")
    print(output_text or json.dumps({"claims": []}))
    return 0


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source", {})
    case = payload.get("case", {})
    return {
        "instructions": payload.get("instructions"),
        "allowed_claim_types": payload.get("allowed_claim_types"),
        "allowed_polarities": payload.get("allowed_polarities"),
        "allowed_strengths": payload.get("allowed_strengths"),
        "case": {
            "chief_complaint": case.get("chief_complaint"),
            "findings": case.get("findings", []),
            "free_text": case.get("free_text", ""),
        },
        "source": {
            "id": source.get("id"),
            "title": source.get("title"),
            "source_type": source.get("source_type"),
            "section_path": source.get("section_path", []),
            "section_type": source.get("section_type"),
            "text": source.get("text", ""),
        },
    }


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "claim_type",
                        "condition",
                        "finding",
                        "polarity",
                        "strength",
                        "exact_quote",
                        "extraction_confidence",
                    ],
                    "properties": {
                        "claim_type": {
                            "type": "string",
                            "enum": [
                                "supports",
                                "argues_against",
                                "requires_test",
                                "rules_in",
                                "rules_out",
                                "contraindicates",
                                "red_flag",
                                "treatment_recommends",
                            ],
                        },
                        "condition": {"type": "string"},
                        "finding": {"type": "string"},
                        "polarity": {
                            "type": "string",
                            "enum": ["supports", "argues_against", "recommends"],
                        },
                        "strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                        "exact_quote": {"type": "string"},
                        "extraction_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            }
        },
    }


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


if __name__ == "__main__":
    raise SystemExit(main())
