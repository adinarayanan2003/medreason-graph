from __future__ import annotations

from medreason_graph.text import normalize


NOISE_SECTION_MARKERS = {
    "authors",
    "affiliations",
    "references",
    "copyright",
    "publication details",
    "bookshelf id",
    "disclosure",
    "conflicts of interest",
}

NOISE_TEXT_MARKERS = (
    "window.",
    "var ",
    "function(",
    "ncbi bookshelf",
    "a service of the national library of medicine",
    "statpearls publishing",
    "bookshelf id:",
    "pmid:",
    "copyright notice",
    "all rights reserved",
)


def is_noise_chunk(text: str, section_path: list[str]) -> bool:
    joined_sections = normalize(" ".join(section_path))
    if any(marker in joined_sections for marker in NOISE_SECTION_MARKERS):
        return True
    lowered = normalize(text)
    if not lowered:
        return True
    if any(marker in lowered for marker in NOISE_TEXT_MARKERS):
        return True
    alpha_ratio = sum(char.isalpha() for char in lowered) / max(len(lowered), 1)
    if len(lowered) > 200 and alpha_ratio < 0.45:
        return True
    return False

