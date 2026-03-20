from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentSettings:
    dataset_id: str
    split: str = "test"
    experiment_name: str = "custom"
    label_card_style: str = "name_only"
    include_full_sequence: bool = False
    model_provider: str = "mock"
    model_name: str = "oracle"
    temperature: float = 0.0
    max_examples: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "experiment_name": self.experiment_name,
            "label_card_style": self.label_card_style,
            "include_full_sequence": self.include_full_sequence,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_examples": self.max_examples,
        }

    def slug(self) -> str:
        parts = [
            self.experiment_name,
            self.dataset_id,
            self.split,
            self.label_card_style,
            f"ctx{int(self.include_full_sequence)}",
            self.model_provider,
            self.model_name.replace("/", "_"),
        ]
        return "__".join(parts)


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    csv_dir: Path
    catalog_path: Path
    track_name: str
    family_code: str
    similarity_split: str

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset_id": self.dataset_id,
            "csv_dir": str(self.csv_dir),
            "catalog_path": str(self.catalog_path),
            "track_name": self.track_name,
            "family_code": self.family_code,
            "similarity_split": self.similarity_split,
        }


@dataclass(frozen=True)
class FragmentExample:
    uid: str
    dataset_id: str
    split: str
    interpro_id: str
    interpro_label: int
    seq_fragment_raw: str
    fragment_parts: tuple[str, ...]
    seq_full: str
    start_parts: tuple[int, ...]
    end_parts: tuple[int, ...]
    is_multi_fragment: bool

    @property
    def fragment_length(self) -> int:
        return sum(len(part) for part in self.fragment_parts)

    def compact_fragment(self) -> str:
        return "".join(self.fragment_parts)

    def ranges(self) -> list[tuple[int, int]]:
        return list(zip(self.start_parts, self.end_parts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "interpro_id": self.interpro_id,
            "interpro_label": self.interpro_label,
            "seq_fragment_raw": self.seq_fragment_raw,
            "fragment_parts": list(self.fragment_parts),
            "seq_full": self.seq_full,
            "start_parts": list(self.start_parts),
            "end_parts": list(self.end_parts),
            "is_multi_fragment": self.is_multi_fragment,
            "fragment_length": self.fragment_length,
        }


@dataclass(frozen=True)
class LabelCard:
    accession: str
    catalog_index: int
    name: str
    label_type: str
    description: str
    go_terms: tuple[str, ...]
    literature_count: int
    short_desc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accession": self.accession,
            "catalog_index": self.catalog_index,
            "name": self.name,
            "label_type": self.label_type,
            "description": self.description,
            "go_terms": list(self.go_terms),
            "literature_count": self.literature_count,
            "short_desc": self.short_desc,
        }


@dataclass(frozen=True)
class ModelResponse:
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Prediction:
    top_ids: tuple[str, ...]
    confidence: float | None
    abstain: bool
    parse_success: bool
    invalid_labels: tuple[str, ...]
    parse_error: str | None = None
    extracted_payload: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_ids": list(self.top_ids),
            "confidence": self.confidence,
            "abstain": self.abstain,
            "parse_success": self.parse_success,
            "invalid_labels": list(self.invalid_labels),
            "parse_error": self.parse_error,
            "extracted_payload": self.extracted_payload,
        }


@dataclass(frozen=True)
class ExampleResult:
    example: FragmentExample
    prompt: str
    raw_response: str
    response_metadata: dict[str, Any]
    prediction: Prediction
    predicted_top_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example": self.example.to_dict(),
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "response_metadata": self.response_metadata,
            "prediction": self.prediction.to_dict(),
            "predicted_top_id": self.predicted_top_id,
        }
