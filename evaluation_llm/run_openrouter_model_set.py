from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from evaluation_llm.model_sets import get_model_set, list_model_sets
from evaluation_llm.records import ExperimentSettings
from evaluation_llm.run_fragment_benchmark import build_experiment_settings, run_single_benchmark


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fragment benchmark across a preset OpenRouter model set")
    parser.add_argument("--dataset_id", required=True, help="Example: VenusX_Res_Act_MF50")
    parser.add_argument("--experiment", default="E2", choices=["E0", "E1", "E2", "E3"])
    parser.add_argument("--split", default="test", choices=["train", "valid", "validation", "test"])
    parser.add_argument("--model_set", default="starter", choices=list_model_sets())
    parser.add_argument("--label_card_style", choices=["name_only", "short_desc", "rich_desc"], default=None)
    parser.add_argument("--include_full_sequence", action="store_true", default=None)
    parser.add_argument("--no_full_sequence", action="store_false", dest="include_full_sequence")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--artifact_root", default="artifacts/evaluation_llm")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    base_args = argparse.Namespace(
        dataset_id=args.dataset_id,
        experiment=args.experiment,
        split=args.split,
        label_card_style=args.label_card_style,
        include_full_sequence=args.include_full_sequence,
        model_provider="openrouter",
        model_name="placeholder",
        temperature=args.temperature,
        max_examples=args.max_examples,
        artifact_root=args.artifact_root,
        suite=False,
    )
    base_settings = build_experiment_settings(base_args)
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    results = []
    for model in get_model_set(args.model_set):
        settings = replace(
            base_settings,
            model_provider="openrouter",
            model_name=model.model_id,
        )
        metrics, run_dir = run_single_benchmark(settings, artifact_root=artifact_root)
        results.append(
            {
                "model_id": model.model_id,
                "family": model.family,
                "tier": model.tier,
                "note": model.note,
                "source_url": model.source_url,
                "metrics": metrics,
                "run_dir": str(run_dir),
            }
        )

    print(json.dumps({"model_set": args.model_set, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
