from __future__ import annotations

import json
import re
from typing import Any

from evaluation_llm.label_catalog import LabelCatalog
from evaluation_llm.records import ExperimentSettings, FragmentExample, LabelCard, Prediction


ACCESSION_RE = re.compile(r"IPR\d+")
MAX_TOP_IDS = 3


def _render_label_card(card: LabelCard, style: str) -> str:
    if style == "name_only":
        return f"- {card.accession} | {card.name}"
    if style == "short_desc":
        return f"- {card.accession} | {card.name} | {card.short_desc}"
    if style == "rich_desc":
        go_terms = ", ".join(card.go_terms[:3]) if card.go_terms else "None"
        return (
            f"- {card.accession} | {card.name}\n"
            f"  Type: {card.label_type}\n"
            f"  Description: {card.description}\n"
            f"  GO terms: {go_terms}\n"
            f"  Literature count: {card.literature_count}"
        )
    raise ValueError(f"Unsupported label_card_style={style!r}")


def _annotate_full_sequence(example: FragmentExample) -> str:
    annotated = example.seq_full
    for index, (start, end) in reversed(list(enumerate(example.ranges(), start=1))):
        if start <= 0 or end < start:
            continue
        left = annotated[: start - 1]
        middle = annotated[start - 1 : end]
        right = annotated[end:]
        annotated = f"{left}<frag{index}:{start}-{end}>{middle}</frag{index}>{right}"
    return annotated


def _render_fragment(example: FragmentExample) -> str:
    parts = []
    for index, (fragment, start, end) in enumerate(
        zip(example.fragment_parts, example.start_parts, example.end_parts),
        start=1,
    ):
        parts.append(f"{index}. residues {start}-{end}: {fragment}")
    return "\n".join(parts)


def build_fragment_prompt(
    example: FragmentExample,
    catalog: LabelCatalog,
    settings: ExperimentSettings,
) -> str:
    sections = [
        "You are doing fragment-level protein function label selection.",
        "Choose the best InterPro accession for the fragment from the candidate labels.",
        'Return JSON only with the schema {"top_ids":["IPR..."],"reasoning_summary":"...", "abstain":false}.',
        f"Use canonical InterPro accessions in top_ids, ordered from best to worst, with at most {MAX_TOP_IDS} candidates.",
        "Keep reasoning_summary short: 1 to 2 sentences explaining why the top label is ranked above the alternatives.",
        f"Experiment: {settings.experiment_name}",
        f"Candidate count: {len(catalog.cards)}",
    ]

    query_lines = [
        f"Query uid: {example.uid}",
        f"Fragment count: {len(example.fragment_parts)}",
        "Fragment parts:",
        _render_fragment(example),
    ]
    if settings.include_full_sequence:
        query_lines.extend(
            [
                f"Full sequence length: {len(example.seq_full)}",
                "Full sequence with fragment tags:",
                _annotate_full_sequence(example),
            ]
        )
    sections.append("\n".join(query_lines))

    rendered_cards = "\n".join(
        _render_label_card(card, settings.label_card_style)
        for card in catalog.sorted_cards()
    )
    sections.append("Candidate labels:\n" + rendered_cards)
    sections.append(
        f"If you are uncertain, set abstain=true and leave top_ids empty. "
        f"If you do answer, return between 1 and {MAX_TOP_IDS} candidate accessions. "
        "If you provide reasoning_summary, keep it concise rather than step-by-step. "
        "Do not invent accessions outside the candidate list."
    )
    return "\n\n".join(sections)


def _extract_json_object(text: str) -> Any | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : index + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_prediction_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, list):
        normalized: list[str] = []
        for item in raw_value:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                for key in ("id", "accession", "label", "name"):
                    if isinstance(item.get(key), str):
                        normalized.append(item[key])
                        break
        return normalized
    return []


def _coerce_bool(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"true", "yes", "1"}
    return False


def _normalize_reasoning_summary(payload: dict[str, Any]) -> str | None:
    for key in ("reasoning_summary", "rationale", "reasoning", "reason", "explanation"):
        value = payload.get(key)
        if isinstance(value, str):
            normalized = " ".join(value.split()).strip()
            if normalized:
                return normalized
    return None


def parse_model_response(raw_text: str, catalog: LabelCatalog) -> Prediction:
    payload = _extract_json_object(raw_text)
    invalid_labels: list[str] = []
    normalized_ids: list[str] = []
    reasoning_summary: str | None = None
    abstain = False

    if isinstance(payload, dict):
        raw_ids = _normalize_prediction_list(
            payload.get("top_ids")
            or payload.get("top_id")
            or payload.get("prediction")
            or payload.get("predictions")
            or payload.get("labels")
        )
        for raw_id in raw_ids:
            accession = catalog.resolve_identifier(raw_id)
            if accession is None:
                invalid_labels.append(raw_id)
            elif accession not in normalized_ids:
                normalized_ids.append(accession)
            if len(normalized_ids) >= MAX_TOP_IDS:
                break

        reasoning_summary = _normalize_reasoning_summary(payload)
        abstain = _coerce_bool(payload.get("abstain"))
        parse_success = True
        parse_error = None
    else:
        accessions = []
        for raw_id in ACCESSION_RE.findall(raw_text.upper()):
            accession = catalog.resolve_identifier(raw_id)
            if accession and accession not in accessions:
                accessions.append(accession)
            if len(accessions) >= MAX_TOP_IDS:
                break
        normalized_ids = accessions
        parse_success = bool(accessions)
        parse_error = None if parse_success else "No valid JSON object or InterPro accession found"
        payload = None
        abstain = "abstain" in raw_text.lower() and not normalized_ids

    if abstain:
        normalized_ids = []

    if not parse_success and not abstain:
        return Prediction(
            top_ids=tuple(),
            reasoning_summary=reasoning_summary,
            abstain=False,
            parse_success=False,
            invalid_labels=tuple(invalid_labels),
            parse_error=parse_error,
            extracted_payload=payload,
        )

    return Prediction(
        top_ids=tuple(normalized_ids),
        reasoning_summary=reasoning_summary,
        abstain=abstain,
        parse_success=True,
        invalid_labels=tuple(invalid_labels),
        parse_error=None,
        extracted_payload=payload,
    )
