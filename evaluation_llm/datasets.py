from __future__ import annotations

import csv
from pathlib import Path

from evaluation_llm.catalog import LabelCatalog
from evaluation_llm.types import DatasetSpec, FragmentExample


SPLIT_FILE_MAP = {
    "train": "train.csv",
    "valid": "valid.csv",
    "validation": "valid.csv",
    "test": "test.csv",
}


def _split_pipe_text(raw: str) -> list[str]:
    return [part.strip() for part in raw.split("|") if part.strip()]


def _split_pipe_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split("|") if part.strip()]


def _build_example(row: dict[str, str], dataset_id: str, split: str) -> FragmentExample:
    fragment_parts = _split_pipe_text(row["seq_fragment"])
    start_parts = _split_pipe_ints(row["start"])
    end_parts = _split_pipe_ints(row["end"])
    if not fragment_parts:
        raise ValueError(f"Row {row['uid']} has no fragment parts")
    if not (len(fragment_parts) == len(start_parts) == len(end_parts)):
        raise ValueError(
            f"Row {row['uid']} has misaligned fragment/start/end values: "
            f"{row['seq_fragment']} | {row['start']} | {row['end']}"
        )

    return FragmentExample(
        uid=row["uid"],
        dataset_id=dataset_id,
        split=split,
        interpro_id=row["interpro_id"],
        interpro_label=int(row["interpro_label"]),
        seq_fragment_raw=row["seq_fragment"],
        fragment_parts=tuple(fragment_parts),
        seq_full=row["seq_full"],
        start_parts=tuple(start_parts),
        end_parts=tuple(end_parts),
        is_multi_fragment=len(fragment_parts) > 1,
    )


def load_fragment_examples(
    spec: DatasetSpec,
    split: str,
    max_examples: int | None = None,
) -> list[FragmentExample]:
    split_key = split.lower()
    if split_key not in SPLIT_FILE_MAP:
        raise ValueError(f"Unsupported split={split!r}. Use one of: {sorted(SPLIT_FILE_MAP)}")

    csv_path = spec.csv_dir / SPLIT_FILE_MAP[split_key]
    if not csv_path.exists():
        raise FileNotFoundError(f"Split file not found: {csv_path}")

    examples: list[FragmentExample] = []
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            examples.append(_build_example(row, dataset_id=spec.dataset_id, split=split_key))
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def inspect_catalog_alignment(examples: list[FragmentExample], catalog: LabelCatalog) -> dict[str, object]:
    missing_accessions: list[str] = []
    inconsistent_labels: dict[int, set[str]] = {}
    label_to_accessions: dict[int, set[str]] = {}
    matches_catalog_index = 0
    mismatches_catalog_index = 0

    for example in examples:
        if example.interpro_id not in catalog.by_accession:
            missing_accessions.append(example.interpro_id)

        label_to_accessions.setdefault(example.interpro_label, set()).add(example.interpro_id)
        card = catalog.by_catalog_index.get(example.interpro_label)
        if card is not None and card.accession == example.interpro_id:
            matches_catalog_index += 1
        else:
            mismatches_catalog_index += 1

    for label, accessions in label_to_accessions.items():
        if len(accessions) > 1:
            inconsistent_labels[label] = accessions

    if missing_accessions:
        missing_preview = ", ".join(sorted(set(missing_accessions))[:5])
        raise ValueError(f"Accessions missing from label catalog: {missing_preview}")
    if inconsistent_labels:
        preview = "\n".join(
            f"{label}: {sorted(accessions)}"
            for label, accessions in list(sorted(inconsistent_labels.items()))[:5]
        )
        raise ValueError(f"Local interpro_label values are inconsistent.\n{preview}")

    return {
        "example_count": len(examples),
        "unique_accessions": len({example.interpro_id for example in examples}),
        "unique_labels": len(label_to_accessions),
        "catalog_index_match_count": matches_catalog_index,
        "catalog_index_mismatch_count": mismatches_catalog_index,
    }


def load_train_label_ids(spec: DatasetSpec) -> set[str]:
    return {example.interpro_id for example in load_fragment_examples(spec, split="train")}


def fragment_length_bin(length: int) -> str:
    if length <= 15:
        return "short"
    if length <= 50:
        return "medium"
    return "long"
