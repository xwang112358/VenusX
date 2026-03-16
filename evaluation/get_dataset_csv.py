"""
Download VenusX datasets from HuggingFace and save each split as a CSV.

Usage:
    # Single dataset
    python evaluation/get_dataset_csv.py \
        --dataset_name AI4Protein/VenusX_Res_Act_MF50

    # All known VenusX datasets
    python evaluation/get_dataset_csv.py --all

    # Custom output directory
    python evaluation/get_dataset_csv.py --all --output_dir /path/to/output
"""

import argparse
import os

from datasets import load_dataset

# Default save location: <repo_root>/data/interpro_2503
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "data", "interpro_2503")

# HuggingFace split name mapping
SPLITS = {"train": "train", "valid": "validation", "test": "test"}

# All known VenusX datasets (Level × Type × Threshold)
_LEVELS = ["Res", "Frag"]
_TYPES = ["Act", "Binding", "Domain", "Motif", "Conserved"]
_THRESHOLDS = ["MF50", "MF70", "MF90"]

KNOWN_DATASETS = [
    f"AI4Protein/VenusX_{level}_{type_}_{thresh}"
    for level in _LEVELS
    for type_ in _TYPES
    for thresh in _THRESHOLDS
]


def save_dataset(dataset_name: str, output_dir: str) -> None:
    short_name = dataset_name.split("/")[-1]  # e.g. VenusX_Res_Act_MF50
    save_dir = os.path.join(output_dir, short_name)
    os.makedirs(save_dir, exist_ok=True)

    for split_name, hf_split in SPLITS.items():
        out_path = os.path.join(save_dir, f"{split_name}.csv")
        print(f"  [{split_name}] loading... ", end="", flush=True)
        ds = load_dataset(dataset_name, split=hf_split)
        df = ds.to_pandas()
        df.to_csv(out_path, index=False)
        print(f"{len(df):,} rows → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download VenusX HuggingFace datasets and save as CSV files"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset_name",
        type=str,
        help="Single HuggingFace dataset ID, e.g. AI4Protein/VenusX_Res_Act_MF50",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all known VenusX datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root directory for CSV output (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    datasets = KNOWN_DATASETS if args.all else [args.dataset_name]

    for i, ds_name in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] {ds_name}")
        try:
            save_dataset(ds_name, args.output_dir)
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
