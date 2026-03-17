from __future__ import annotations

import json
import re
from typing import Any

from evaluation_llm.catalog import LabelCatalog
from evaluation_llm.interfaces import ResponseParser
from evaluation_llm.types import ParsedPrediction


ACCESSION_RE = re.compile(r"IPR\d+")


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


def _normalize_candidate_list(raw_value: Any) -> list[str]:
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


class JsonResponseParser(ResponseParser):
    def parse(self, raw_text: str, catalog: LabelCatalog) -> ParsedPrediction:
        payload = _extract_json_object(raw_text)
        invalid_labels: list[str] = []
        normalized_ids: list[str] = []
        confidence: float | None = None
        abstain = False

        if isinstance(payload, dict):
            raw_ids = _normalize_candidate_list(
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

            raw_confidence = payload.get("confidence")
            if isinstance(raw_confidence, (int, float)):
                confidence = max(0.0, min(1.0, float(raw_confidence)))
            abstain = _coerce_bool(payload.get("abstain"))
            parse_success = True
            parse_error = None
        else:
            accessions = []
            for raw_id in ACCESSION_RE.findall(raw_text.upper()):
                accession = catalog.resolve_identifier(raw_id)
                if accession and accession not in accessions:
                    accessions.append(accession)
            normalized_ids = accessions
            parse_success = bool(accessions)
            parse_error = None if parse_success else "No valid JSON object or InterPro accession found"
            payload = None
            abstain = "abstain" in raw_text.lower() and not normalized_ids

        if abstain:
            normalized_ids = []

        if not parse_success and not abstain:
            return ParsedPrediction(
                top_ids=tuple(),
                confidence=confidence,
                abstain=False,
                parse_success=False,
                invalid_labels=tuple(invalid_labels),
                parse_error=parse_error,
                extracted_payload=payload,
            )

        return ParsedPrediction(
            top_ids=tuple(normalized_ids),
            confidence=confidence,
            abstain=abstain,
            parse_success=True,
            invalid_labels=tuple(invalid_labels),
            parse_error=None,
            extracted_payload=payload,
        )
