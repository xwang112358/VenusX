from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_utils import load_default_env_file
from evaluation_agent.dataset import load_examples
from evaluation_agent.metrics import fragment_metrics, residue_metrics


DEFAULT_CSV_PATH = Path("data/interpro_2503/VenusX_Res_Act_MF50/test.csv")


def parse_args() -> argparse.Namespace:
    load_default_env_file()

    parser = argparse.ArgumentParser(
        description=(
            "Run ProteinAgent on the first instance of a VenusX CSV split and "
            "compare the returned label/location against the gold annotation."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the VenusX CSV file (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("INTERPROSCAN_EMAIL", ""),
        help="Email required by InterProScan. Falls back to INTERPROSCAN_EMAIL.",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Anthropic model name passed to ProteinAgent.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between InterProScan status polls.",
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        default=300,
        help="Maximum seconds to wait for InterProScan.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds for InterProScan calls.",
    )
    return parser.parse_args()


def residues_from_parts(parts: tuple[tuple[int, int], ...]) -> frozenset[int]:
    residues: set[int] = set()
    for start, end in parts:
        residues.update(range(start, end + 1))
    return frozenset(residues)


def extract_fragment(sequence: str, parts: tuple[tuple[int, int], ...]) -> str:
    return "|".join(sequence[start - 1:end] for start, end in parts)


def main() -> None:
    args = parse_args()
    if not args.email:
        raise SystemExit("ERROR: --email is required (or set INTERPROSCAN_EMAIL in .env).")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ERROR: ANTHROPIC_API_KEY is required (set it in env or .env).")

    from protein_agent.agent import ProteinAgent

    examples = load_examples(args.csv)
    example = examples[0]

    gold_parts = example.fragment_parts
    gold_residues = residues_from_parts(gold_parts)
    gold_fragment = extract_fragment(example.seq_full, gold_parts)

    agent = ProteinAgent(
        email=args.email,
        model=args.model,
        poll_interval=args.poll_interval,
        max_wait=args.max_wait,
        timeout=args.timeout,
    )
    result = agent.run(example.seq_full)

    matching_annotations = [
        ann for ann in result.site_annotations if ann.accession == example.interpro_id
    ]
    label_found = bool(matching_annotations)

    print("=== Gold example ===")
    print(f"csv_path: {args.csv}")
    print(f"uid: {example.uid}")
    print(f"gold_interpro_id: {example.interpro_id}")
    print(f"gold_parts: {gold_parts}")
    print(f"gold_fragment: {gold_fragment}")
    print(f"sequence_length: {len(example.seq_full)}")
    print()

    print("=== Agent summary ===")
    print(f"model: {result.metadata.get('model')}")
    print(f"tool_calls: {result.metadata.get('tool_calls')}")
    print(f"usage: {result.metadata.get('usage')}")
    print(f"n_annotations: {len(result.annotations)}")
    print(f"n_site_annotations: {len(result.site_annotations)}")
    print(f"label_found: {label_found}")
    print()

    if matching_annotations:
        print("=== Matching annotation(s) ===")
        for idx, ann in enumerate(matching_annotations, start=1):
            pred_residues = ann.residue_set()
            pred_fragment_metrics = fragment_metrics(gold_parts, ann.locations)
            pred_residue_metrics = residue_metrics(gold_residues, pred_residues)
            print(
                f"[{idx}] accession={ann.accession} name={ann.name} "
                f"site_type={ann.site_type} locations={ann.locations}"
            )
            print(f"    residue_metrics={pred_residue_metrics}")
            print(f"    fragment_metrics={pred_fragment_metrics}")
    else:
        print("=== Returned site annotations ===")
        for idx, ann in enumerate(result.site_annotations, start=1):
            print(
                f"[{idx}] accession={ann.accession} name={ann.name} "
                f"site_type={ann.site_type} locations={ann.locations}"
            )


if __name__ == "__main__":
    main()
