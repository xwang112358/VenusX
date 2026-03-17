from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from evaluation_llm.types import LabelCard


HTML_TAG_RE = re.compile(r"<[^>]+>")
CITATION_RE = re.compile(r"\[\[cite:[^\]]+\]\]")
WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = CITATION_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"(?<=[.!?])\s", text)
    if match:
        return text[: match.start()].strip()
    return text.strip()


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + "..."


def _go_term_names(go_terms: list[dict]) -> tuple[str, ...]:
    names = []
    for item in go_terms or []:
        name = item.get("name")
        if name:
            names.append(name)
    return tuple(names)


def build_short_desc(description: str, go_term_names: tuple[str, ...], max_words: int = 40) -> str:
    first_sentence = _first_sentence(_normalize_text(description))
    summary = _truncate_words(first_sentence, max_words=max_words)
    selected_go_terms = [name for name in go_term_names[:2] if name.lower() not in summary.lower()]
    if selected_go_terms:
        go_suffix = "; GO: " + ", ".join(selected_go_terms)
        summary = f"{summary}{go_suffix}" if summary else go_suffix.lstrip("; ")
    return summary


@dataclass(frozen=True)
class LabelCatalog:
    cards: tuple[LabelCard, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_accession", {card.accession: card for card in self.cards})
        object.__setattr__(self, "by_catalog_index", {card.catalog_index: card for card in self.cards})

        by_name: dict[str, list[LabelCard]] = {}
        for card in self.cards:
            by_name.setdefault(normalize_label_name(card.name), []).append(card)
        object.__setattr__(
            self,
            "by_name",
            {name: tuple(sorted(matches, key=lambda card: card.accession)) for name, matches in by_name.items()},
        )

    def resolve_identifier(self, raw_value: str) -> str | None:
        value = raw_value.strip()
        if not value:
            return None
        accession_match = re.search(r"(IPR\d+)", value.upper())
        if accession_match:
            accession = accession_match.group(1)
            if accession in self.by_accession:
                return accession

        matches = self.by_name.get(normalize_label_name(value))
        if matches and len(matches) == 1:
            return matches[0].accession
        return None

    def cards_for_accessions(self, accessions: list[str]) -> tuple[LabelCard, ...]:
        cards = [self.by_accession[accession] for accession in accessions if accession in self.by_accession]
        return tuple(sorted(cards, key=lambda card: card.accession))

    def sorted_cards(self) -> tuple[LabelCard, ...]:
        return tuple(sorted(self.cards, key=lambda card: card.accession))

    def to_dict(self) -> dict:
        return {"cards": [card.to_dict() for card in self.cards]}


def normalize_label_name(name: str) -> str:
    return WHITESPACE_RE.sub(" ", name.strip().lower())


def load_label_catalog(path: str | Path) -> LabelCatalog:
    path = Path(path)
    raw_items = json.loads(path.read_text())
    cards: list[LabelCard] = []
    for index, item in enumerate(raw_items):
        metadata = item["metadata"]
        description = _normalize_text(metadata.get("description") or "")
        go_terms = _go_term_names(metadata.get("go_terms") or [])
        literature = metadata.get("literature") or {}
        cards.append(
            LabelCard(
                accession=metadata["accession"],
                catalog_index=index,
                name=metadata.get("name") or metadata["accession"],
                label_type=metadata.get("type") or "",
                description=description,
                go_terms=go_terms,
                literature_count=len(literature),
                short_desc=build_short_desc(description, go_terms),
            )
        )
    return LabelCatalog(cards=tuple(cards))
