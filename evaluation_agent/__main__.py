"""CLI entry point for the agent evaluation framework.

Usage
-----
python -m evaluation_agent \\
    --csv  data/interpro_2503/VenusX_Res_Act_MF50/test.csv \\
    --email  you@example.com \\
    [--model  gpt-4o] \\
    [--max_examples  50] \\
    [--iou_threshold  0.5] \\
    [--out  results.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from env_utils import load_default_env_file
from evaluation_agent.metrics import aggregate_results
from evaluation_agent.runner import run_evaluation


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_default_env_file()

    parser = argparse.ArgumentParser(
        prog="python -m evaluation_agent",
        description="Evaluate the ProteinAgent on a VenusX CSV dataset.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        metavar="PATH",
        help="Path to a VenusX split CSV (train/valid/test).",
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("INTERPROSCAN_EMAIL", ""),
        metavar="ADDR",
        help=(
            "E-mail address for EBI InterProScan (required by EBI terms of service). "
            "Falls back to INTERPROSCAN_EMAIL env var."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        metavar="ID",
        help="Model ID passed to ProteinAgent (default: gpt-4o).",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N examples (useful for smoke tests).",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="IoU threshold for fragment matching (default: 0.5).",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Write per-example results as JSONL to this file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if not args.email:
        print(
            "ERROR: --email is required (or set INTERPROSCAN_EMAIL).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is required (or set it in .env).",
            file=sys.stderr,
        )
        sys.exit(1)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    from protein_agent.agent import ProteinAgent

    agent = ProteinAgent(
        email=args.email,
        model=args.model,
    )

    results = run_evaluation(
        csv_path=csv_path,
        agent=agent,
        max_examples=args.max_examples,
        iou_threshold=args.iou_threshold,
    )

    # Write per-example JSONL
    if args.out:
        out_path = Path(args.out)
        with out_path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(asdict(r)) + "\n")
        print(f"\nPer-example results written to {out_path}")

    # Print aggregate summary table
    summary = aggregate_results(results)
    print("\n" + "=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)
    print(f"  Examples evaluated : {summary['n_total']}")
    print(f"  Errors             : {summary['n_errors']}")
    print(f"  Label recall       : {summary['label_recall']:.3f}"
          f"  ({summary['n_label_found']}/{summary['n_total']})")
    print()
    print("  Residue-level  (on label-found examples)")
    print(f"    Precision : {_fmt(summary['mean_residue_precision'])}")
    print(f"    Recall    : {_fmt(summary['mean_residue_recall'])}")
    print(f"    F1        : {_fmt(summary['mean_residue_f1'])}")
    print()
    print(f"  Fragment-level (IoU ≥ {args.iou_threshold})")
    print(f"    Precision : {_fmt(summary['mean_fragment_precision'])}")
    print(f"    Recall    : {_fmt(summary['mean_fragment_recall'])}")
    print(f"    F1        : {_fmt(summary['mean_fragment_f1'])}")
    print("=" * 60)


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


if __name__ == "__main__":
    main()
