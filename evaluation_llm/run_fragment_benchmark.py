from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from evaluation_llm.fragment_dataset import (
    get_dataset_info,
    load_fragment_examples,
    summarize_catalog_alignment,
)
from evaluation_llm.label_catalog import load_label_catalog
from evaluation_llm.metrics import FragmentBenchmarkMetrics
from evaluation_llm.model_backends import create_model_backend
from evaluation_llm.prompt_and_parse import build_fragment_prompt, parse_model_response
from evaluation_llm.records import ExampleResult, ExperimentSettings, Prediction


EXPERIMENT_PRESETS = {
    "E0": {
        "experiment_name": "E0_Smoke",
        "label_card_style": "name_only",
        "include_full_sequence": False,
        "model_provider": "mock",
        "model_name": "oracle",
        "max_examples": 10,
    },
    "E1": {
        "experiment_name": "E1_FullCatalogName",
        "label_card_style": "name_only",
        "include_full_sequence": False,
    },
    "E2": {
        "experiment_name": "E2_FullCatalogShortDesc",
        "label_card_style": "short_desc",
        "include_full_sequence": False,
    },
    "E3": {
        "experiment_name": "E3_FullCatalogContext",
        "label_card_style": "short_desc",
        "include_full_sequence": True,
    },
}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default) + "\n")


def default_model_args() -> dict[str, object]:
    return {
        "model_provider": "mock",
        "model_name": "heuristic",
    }


def build_experiment_settings(args: argparse.Namespace) -> ExperimentSettings:
    preset = EXPERIMENT_PRESETS[args.experiment]
    settings = ExperimentSettings(
        dataset_id=args.dataset_id,
        split=args.split,
        experiment_name=preset["experiment_name"],
        label_card_style=preset["label_card_style"],
        include_full_sequence=preset["include_full_sequence"],
        model_provider=args.model_provider,
        model_name=args.model_name,
        temperature=args.temperature,
        max_examples=args.max_examples,
    )

    if preset.get("model_provider") and args.model_provider == default_model_args()["model_provider"]:
        settings = replace(settings, model_provider=preset["model_provider"])
    if preset.get("model_name") and args.model_name == default_model_args()["model_name"]:
        settings = replace(settings, model_name=preset["model_name"])
    if preset.get("max_examples") is not None and args.max_examples is None:
        settings = replace(settings, max_examples=preset["max_examples"])

    if args.label_card_style is not None:
        settings = replace(settings, label_card_style=args.label_card_style)
    if args.include_full_sequence is not None:
        settings = replace(settings, include_full_sequence=args.include_full_sequence)
    return settings


def _resolve_run_dir(artifact_root: Path, settings: ExperimentSettings) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifact_root / settings.dataset_id / f"{settings.slug()}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_single_benchmark(
    settings: ExperimentSettings,
    artifact_root: Path,
) -> tuple[dict, Path]:
    dataset_info = get_dataset_info(settings.dataset_id)
    catalog = load_label_catalog(dataset_info.catalog_path)
    label_cards = catalog.sorted_cards()
    examples = load_fragment_examples(dataset_info, split=settings.split, max_examples=settings.max_examples)
    alignment_summary = summarize_catalog_alignment(examples, catalog)

    backend = create_model_backend(
        settings.model_provider,
        settings.model_name,
        catalog,
        temperature=settings.temperature,
    )
    metrics = FragmentBenchmarkMetrics()
    run_dir = _resolve_run_dir(artifact_root=artifact_root, settings=settings)
    _write_json(
        run_dir / "resolved_config.json",
        {
            "settings": settings.to_dict(),
            "dataset_info": dataset_info.to_dict(),
            "alignment_summary": alignment_summary,
        },
    )

    results: list[ExampleResult] = []
    errors: list[dict] = []
    for example in tqdm(
        examples,
        desc=f"{settings.experiment_name}:{settings.split}",
        leave=False,
        disable=not sys.stderr.isatty(),
    ):
        prompt = ""
        raw_response = ""
        response_metadata: dict[str, object] = {"mode": settings.model_provider}
        try:
            prompt = build_fragment_prompt(example, catalog, settings)
            response = backend.generate(prompt=prompt, example=example, label_cards=label_cards)
            raw_response = response.raw_text
            response_metadata = response.metadata
            prediction = parse_model_response(response.raw_text, catalog)
        except Exception as exc:
            prediction = Prediction(
                top_ids=tuple(),
                confidence=None,
                abstain=False,
                parse_success=False,
                invalid_labels=tuple(),
                parse_error=f"backend_error: {type(exc).__name__}: {exc}",
                extracted_payload=None,
            )
            response_metadata = {
                **response_metadata,
                "backend_error": f"{type(exc).__name__}: {exc}",
            }
        predicted_top_id = prediction.top_ids[0] if prediction.top_ids else None
        result = ExampleResult(
            example=example,
            prompt=prompt,
            raw_response=raw_response,
            response_metadata=response_metadata,
            prediction=prediction,
            predicted_top_id=predicted_top_id,
        )
        results.append(result)
        metrics.update(result)
        if not prediction.parse_success or prediction.invalid_labels or "backend_error" in response_metadata:
            errors.append(result.to_dict())

    summary = metrics.compute()
    _write_json(run_dir / "metrics.json", summary)
    _write_jsonl(run_dir / "records.jsonl", (result.to_dict() for result in results))
    _write_jsonl(run_dir / "errors.jsonl", errors)
    return summary, run_dir


def _score_summary(metrics: dict) -> tuple:
    paper = metrics["main_paper_table"]
    return (
        paper.get("accuracy") or 0.0,
        paper.get("macro_f1") or 0.0,
        paper.get("mcc") or 0.0,
        paper.get("macro_precision") or 0.0,
        paper.get("macro_recall") or 0.0,
    )


def _related_dataset_ids(dataset_id: str) -> list[str]:
    if not dataset_id.endswith("MF50"):
        raise ValueError("Suite mode expects an MF50 dataset_id so it can project to MF70 and MF90.")
    return [dataset_id, dataset_id.replace("MF50", "MF70"), dataset_id.replace("MF50", "MF90")]


def build_suite_settings(base_settings: ExperimentSettings) -> list[ExperimentSettings]:
    return [
        replace(base_settings, experiment_name=EXPERIMENT_PRESETS["E1"]["experiment_name"], label_card_style="name_only", include_full_sequence=False),
        replace(base_settings, experiment_name=EXPERIMENT_PRESETS["E2"]["experiment_name"], label_card_style="short_desc", include_full_sequence=False),
        replace(base_settings, experiment_name=EXPERIMENT_PRESETS["E3"]["experiment_name"], label_card_style="short_desc", include_full_sequence=True),
    ]


def run_selection_suite(
    base_settings: ExperimentSettings,
    artifact_root: Path,
) -> tuple[dict, Path]:
    suite_dir = _resolve_run_dir(artifact_root=artifact_root, settings=replace(base_settings, experiment_name="suite"))
    validation_settings = replace(base_settings, split="valid")
    validation_runs = []
    best_run = None

    for settings in build_suite_settings(validation_settings):
        metrics, run_dir = run_single_benchmark(settings, artifact_root=suite_dir / "runs")
        result = {
            "settings": settings.to_dict(),
            "metrics": metrics,
            "run_dir": str(run_dir),
        }
        validation_runs.append(result)
        if best_run is None or _score_summary(metrics) > _score_summary(best_run["metrics"]):
            best_run = result

    assert best_run is not None
    selected_settings = ExperimentSettings(**best_run["settings"])
    test_runs = []
    for dataset_id in _related_dataset_ids(base_settings.dataset_id):
        settings = replace(selected_settings, dataset_id=dataset_id, split="test")
        metrics, run_dir = run_single_benchmark(settings, artifact_root=suite_dir / "selected")
        test_runs.append(
            {
                "settings": settings.to_dict(),
                "metrics": metrics,
                "run_dir": str(run_dir),
            }
        )

    summary = {
        "validation_runs": validation_runs,
        "selected_settings": selected_settings.to_dict(),
        "selected_validation_metrics": best_run["metrics"],
        "test_runs": test_runs,
    }
    _write_json(suite_dir / "suite_summary.json", summary)
    return summary, suite_dir


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fragment-level LLM benchmark runner")
    parser.add_argument("--dataset_id", required=True, help="Example: VenusX_Res_Act_MF50")
    parser.add_argument("--experiment", default="E0", choices=sorted(EXPERIMENT_PRESETS))
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--label_card_style", choices=["name_only", "short_desc", "rich_desc"], default=None)
    parser.add_argument("--include_full_sequence", action="store_true", default=None)
    parser.add_argument("--no_full_sequence", action="store_false", dest="include_full_sequence")
    parser.add_argument("--model_provider", default=default_model_args()["model_provider"], choices=["mock", "replay", "openrouter", "agent"])
    parser.add_argument("--model_name", default=default_model_args()["model_name"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--artifact_root", default="artifacts/evaluation_llm")
    parser.add_argument("--suite", action="store_true", help="Run E1/E2/E3 on validation, then test MF50/MF70/MF90")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    settings = build_experiment_settings(args)
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    if args.suite:
        summary, suite_dir = run_selection_suite(settings, artifact_root=artifact_root)
        print(json.dumps({"suite_dir": str(suite_dir), "selected_settings": summary["selected_settings"]}, indent=2))
        return 0

    metrics, run_dir = run_single_benchmark(settings, artifact_root=artifact_root)
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
