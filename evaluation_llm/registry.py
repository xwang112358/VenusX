from __future__ import annotations

import re
from pathlib import Path

from evaluation_llm.types import DatasetSpec


DATASET_PATTERN = re.compile(r"^(VenusX_Res_(Act|BindI)_(MF50|MF70|MF90))$")
TRACK_BY_FAMILY = {
    "Act": "active_site",
    "BindI": "binding_site",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def list_supported_dataset_ids() -> list[str]:
    dataset_ids: list[str] = []
    for family_code in TRACK_BY_FAMILY:
        for similarity_split in ("MF50", "MF70", "MF90"):
            dataset_ids.append(f"VenusX_Res_{family_code}_{similarity_split}")
    return dataset_ids


def get_dataset_spec(dataset_id: str, root: Path | None = None) -> DatasetSpec:
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

    return DatasetSpec(
        dataset_id=dataset_id,
        csv_dir=csv_dir,
        catalog_path=catalog_path,
        track_name=track_name,
        family_code=family_code,
        similarity_split=similarity_split,
    )
