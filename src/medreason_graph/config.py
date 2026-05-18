from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medreason_graph import lexicon
from medreason_graph.lexicon import Concept


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default_clinical_config.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else default_config_path()
    return json.loads(config_path.read_text(encoding="utf-8"))


def apply_config(config: dict[str, Any]) -> None:
    if "concepts" in config:
        lexicon.CONCEPTS.clear()
        for canonical, payload in config["concepts"].items():
            lexicon.CONCEPTS[canonical] = Concept(
                canonical=canonical,
                kind=payload["kind"],
                synonyms=tuple(payload.get("synonyms", [canonical])),
                semantic_type=payload.get("semantic_type"),
            )
    _replace_mapping(lexicon.SOURCE_TYPE_WEIGHT, config.get("source_type_weight"))
    _replace_mapping(lexicon.STRENGTH_WEIGHT, config.get("strength_weight"))
    _replace_mapping(lexicon.HIGH_RISK_CONDITIONS, config.get("high_risk_conditions"))
    _replace_mapping(lexicon.DANGEROUS_ALTERNATIVES, config.get("dangerous_alternatives"), tuple_values=True)
    _replace_mapping(lexicon.SECTION_KEYWORDS, config.get("section_keywords"), tuple_values=True)
    _replace_mapping(lexicon.AMBIGUOUS_ABBREVIATIONS, config.get("ambiguous_abbreviations"), tuple_values=True)
    _replace_nested_tuple_mapping(lexicon.ABBREVIATION_CONTEXT, config.get("abbreviation_context"))


def load_and_apply_config(path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(path)
    apply_config(config)
    return config


def _replace_mapping(target: dict, incoming: dict[str, Any] | None, *, tuple_values: bool = False) -> None:
    if incoming is None:
        return
    target.clear()
    for key, value in incoming.items():
        target[key] = tuple(value) if tuple_values else value


def _replace_nested_tuple_mapping(target: dict, incoming: dict[str, Any] | None) -> None:
    if incoming is None:
        return
    target.clear()
    for key, value in incoming.items():
        target[key] = {nested_key: tuple(nested_value) for nested_key, nested_value in value.items()}

