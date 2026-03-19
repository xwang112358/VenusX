from __future__ import annotations

import csv
import re
from pathlib import Path

from evaluation_llm.label_catalog import LabelCatalog
from evaluation_llm.records import DatasetInfo, FragmentExample


DATASET_PATTERN = re.compile(r"^(VenusX_Res_(Act|BindI)_(MF50|MF70|MF90))$")
TRACK_BY_FAMILY = {
    "Act": "active_site",
    "BindI": "binding_site",
}
SPLIT_FILE_MAP = {
    "train": "train.csv",
    "valid": "valid.csv",
    "validation": "valid.csv",
    "test": "test.csv",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def list_supported_dataset_ids() -> list[str]:
    dataset_ids: list[str] = []
    for family_code in TRACK_BY_FAMILY:
        for similarity_split in ("MF50", "MF70", "MF90"):
            dataset_ids.append(f"VenusX_Res_{family_code}_{similarity_split}")
    return dataset_ids


def get_dataset_info(dataset_id: str, root: Path | None = None) -> DatasetInfo:
    root = root or repo_root()
    match = DATASET_PATTERN.match(dataset_id)
    if match is None:
        raise ValueError(
            f"Unsupported dataset_id={dataset_id!r}. "
            f"Supported ids: {', '.join(list_supported_dataset_ids())}"
        )

    _, family_code, similarity_split = match.groups()
    track_name = TRACK_BY_FAMILY[family_code]
    csv_dir = root / "data" / "interpro_2503" / dataset_id
    catalog_path = root / "data" / "interpro_2503" / track_name / f"{track_name}_des.json"
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    return DatasetInfo(
        dataset_id=dataset_id,
        csv_dir=csv_dir,
        catalog_path=catalog_path,
        track_name=track_name,
        family_code=family_code,
        similarity_split=similarity_split,
    )


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
    dataset_info: DatasetInfo,
    split: str,
    max_examples: int | None = None,
) -> list[FragmentExample]:
    split_key = split.lower()
    if split_key not in SPLIT_FILE_MAP:
        raise ValueError(f"Unsupported split={split!r}. Use one of: {sorted(SPLIT_FILE_MAP)}")

    csv_path = dataset_info.csv_dir / SPLIT_FILE_MAP[split_key]
    if not csv_path.exists():
        raise FileNotFoundError(f"Split file not found: {csv_path}")

    examples: list[FragmentExample] = []
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            examples.append(_build_example(row, dataset_id=dataset_info.dataset_id, split=split_key))
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def summarize_catalog_alignment(examples: list[FragmentExample], catalog: LabelCatalog) -> dict[str, object]:
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


def load_train_label_ids(dataset_info: DatasetInfo) -> set[str]:
    return {example.interpro_id for example in load_fragment_examples(dataset_info, split="train")}


def fragment_length_bin(length: int) -> str:
    if length <= 15:
        return "short"
    if length <= 50:
        return "medium"
    return "long"
