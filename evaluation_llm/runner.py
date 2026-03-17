from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from evaluation_llm.adapters import MockModelAdapter, PlaceholderAgentAdapter, ReplayModelAdapter
from evaluation_llm.catalog import LabelCatalog, load_label_catalog
from evaluation_llm.datasets import (
    inspect_catalog_alignment,
    load_fragment_examples,
)
from evaluation_llm.interfaces import CandidateProvider
from evaluation_llm.metrics import FragmentMetricSuite
from evaluation_llm.parsing import JsonResponseParser
from evaluation_llm.prompting import FragmentPromptBuilder
from evaluation_llm.registry import get_dataset_spec
from evaluation_llm.retrieval import FullCatalogCandidateProvider, TopKPrototypeCandidateProvider
from evaluation_llm.types import EvaluationRecord, PromptContext, RunConfig


EXPERIMENT_PRESETS = {
    "E0": {
        "experiment_name": "E0_Smoke",
        "candidate_strategy": "full_catalog",
        "label_card_style": "name_only",
        "include_full_sequence": False,
        "few_shot_count": 0,
        "model_provider": "mock",
        "model_name": "oracle",
        "max_examples": 10,
    },
    "E1": {
        "experiment_name": "E1_FullCatalogName",
        "candidate_strategy": "full_catalog",
        "label_card_style": "name_only",
        "include_full_sequence": False,
        "few_shot_count": 0,
    },
    "E2": {
        "experiment_name": "E2_FullCatalogShortDesc",
        "candidate_strategy": "full_catalog",
        "label_card_style": "short_desc",
        "include_full_sequence": False,
        "few_shot_count": 0,
    },
    "E3": {
        "experiment_name": "E3_FullCatalogContext",
        "candidate_strategy": "full_catalog",
        "label_card_style": "short_desc",
        "include_full_sequence": True,
        "few_shot_count": 0,
    },
    "E4": {
        "experiment_name": "E4_TopKPrototypeRerank",
        "candidate_strategy": "topk_prototype",
        "label_card_style": "short_desc",
        "include_full_sequence": False,
        "few_shot_count": 0,
    },
    "E5": {
        "experiment_name": "E5_TopKPrototypeRerankContext",
        "candidate_strategy": "topk_prototype",
        "label_card_style": "short_desc",
        "include_full_sequence": True,
        "few_shot_count": 0,
    },
    "E6": {
        "experiment_name": "E6_FewShotRerank",
        "candidate_strategy": "topk_prototype",
        "label_card_style": "short_desc",
        "include_full_sequence": False,
        "few_shot_count": 2,
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


def build_run_config(args: argparse.Namespace) -> RunConfig:
    preset = EXPERIMENT_PRESETS[args.experiment]
    base = RunConfig(
        dataset_id=args.dataset_id,
        split=args.split,
        label_space="open_catalog",
        candidate_strategy=preset["candidate_strategy"],
        top_k=args.top_k,
        prompt_template=args.prompt_template,
        label_card_style=preset["label_card_style"],
        include_full_sequence=preset["include_full_sequence"],
        few_shot_count=preset["few_shot_count"],
        model_provider=args.model_provider,
        model_name=args.model_name,
        temperature=args.temperature,
        max_examples=args.max_examples,
        experiment_name=preset["experiment_name"],
    )

    if preset.get("model_provider") and args.model_provider == parser_defaults()["model_provider"]:
        base = replace(base, model_provider=preset["model_provider"])
    if preset.get("model_name") and args.model_name == parser_defaults()["model_name"]:
        base = replace(base, model_name=preset["model_name"])
    if preset.get("max_examples") is not None and args.max_examples is None:
        base = replace(base, max_examples=preset["max_examples"])

    if args.label_card_style is not None:
        base = replace(base, label_card_style=args.label_card_style)
    if args.include_full_sequence is not None:
        base = replace(base, include_full_sequence=args.include_full_sequence)
    if args.few_shot_count is not None:
        base = replace(base, few_shot_count=args.few_shot_count)
    if args.candidate_strategy is not None:
        base = replace(base, candidate_strategy=args.candidate_strategy)
    return base


def parser_defaults() -> dict[str, object]:
    return {
        "model_provider": "mock",
        "model_name": "heuristic",
    }


def _instantiate_candidate_provider(
    config: RunConfig,
    catalog: LabelCatalog,
    train_examples,
) -> CandidateProvider:
    if config.candidate_strategy == "full_catalog":
        return FullCatalogCandidateProvider(catalog)
    if config.candidate_strategy == "topk_prototype":
        return TopKPrototypeCandidateProvider(catalog=catalog, train_examples=train_examples)
    raise ValueError(f"Unsupported candidate_strategy={config.candidate_strategy!r}")


def _instantiate_model_adapter(config: RunConfig, catalog: LabelCatalog):
    if config.model_provider == "mock":
        return MockModelAdapter(catalog=catalog, mode=config.model_name)
    if config.model_provider == "replay":
        return ReplayModelAdapter(config.model_name)
    if config.model_provider == "agent":
        return PlaceholderAgentAdapter()
    raise ValueError(f"Unsupported model_provider={config.model_provider!r}")


def _resolve_artifact_dir(artifact_root: Path, config: RunConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifact_root / config.dataset_id / f"{config.slug()}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_single(config: RunConfig, artifact_root: Path) -> tuple[dict, Path]:
    spec = get_dataset_spec(config.dataset_id)
    catalog = load_label_catalog(spec.catalog_path)
    examples = load_fragment_examples(spec, split=config.split, max_examples=config.max_examples)
    alignment_summary = inspect_catalog_alignment(examples, catalog)
    train_examples = load_fragment_examples(spec, split="train")
    train_label_ids = {example.interpro_id for example in train_examples}

    candidate_provider = _instantiate_candidate_provider(config, catalog, train_examples)
    model_adapter = _instantiate_model_adapter(config, catalog)
    prompt_builder = FragmentPromptBuilder()
    parser = JsonResponseParser()
    metrics = FragmentMetricSuite()

    run_dir = _resolve_artifact_dir(artifact_root=artifact_root, config=config)
    _write_json(
        run_dir / "resolved_config.json",
        {
            "config": config.to_dict(),
            "dataset_spec": spec.to_dict(),
            "alignment_summary": alignment_summary,
        },
    )

    records: list[EvaluationRecord] = []
    parser_failures: list[dict] = []
    for example in tqdm(
        examples,
        desc=f"{config.experiment_name}:{config.split}",
        leave=False,
        disable=not sys.stderr.isatty(),
    ):
        candidates = candidate_provider.get_candidates(example, top_k=config.top_k)
        candidate_ids = [candidate.accession for candidate in candidates]
        candidate_cards = catalog.cards_for_accessions(candidate_ids)
        few_shots = candidate_provider.get_few_shots(
            example,
            candidate_ids=candidate_ids,
            limit=config.few_shot_count,
        )
        context = PromptContext(
            example=example,
            config=config,
            candidate_records=tuple(candidates),
            candidate_cards=candidate_cards,
            few_shot_examples=tuple(few_shots),
        )
        package = prompt_builder.build(context=context, catalog=catalog)
        response = model_adapter.generate(package=package, context=context)
        parsed = parser.parse(response.raw_text, catalog=catalog)
        predicted_top_id = parsed.top_ids[0] if parsed.top_ids else None
        candidate_hit = None
        prediction_in_candidates = None
        if config.candidate_strategy != "full_catalog":
            candidate_hit = example.interpro_id in candidate_ids
            prediction_in_candidates = predicted_top_id in set(candidate_ids) if predicted_top_id else False

        record = EvaluationRecord(
            example=example,
            candidates=tuple(candidates),
            parsed=parsed,
            prompt=package.prompt,
            raw_response=response.raw_text,
            response_metadata=response.metadata,
            seen_in_train=example.interpro_id in train_label_ids,
            predicted_top_id=predicted_top_id,
            candidate_hit=candidate_hit,
            prediction_in_candidates=prediction_in_candidates,
        )
        records.append(record)
        metrics.update(record)
        if not parsed.parse_success or parsed.invalid_labels:
            parser_failures.append(record.to_dict())

    summary = metrics.compute()
    _write_json(run_dir / "metrics.json", summary)
    _write_jsonl(run_dir / "records.jsonl", (record.to_dict() for record in records))
    _write_jsonl(run_dir / "errors.jsonl", parser_failures)

    return summary, run_dir


def _config_score(metrics: dict) -> tuple:
    overall = metrics["overall"]
    return (
        overall.get("top1_acc") or 0.0,
        overall.get("macro_f1") or 0.0,
        overall.get("top5_acc") or 0.0,
        overall.get("parse_success_rate") or 0.0,
    )


def _related_dataset_ids(dataset_id: str) -> list[str]:
    if not dataset_id.endswith("MF50"):
        raise ValueError("Suite mode expects an MF50 dataset_id so it can project to MF70 and MF90.")
    return [dataset_id, dataset_id.replace("MF50", "MF70"), dataset_id.replace("MF50", "MF90")]


def build_suite_configs(base_config: RunConfig) -> list[RunConfig]:
    configs = [
        replace(base_config, experiment_name=EXPERIMENT_PRESETS["E1"]["experiment_name"], candidate_strategy="full_catalog", label_card_style="name_only", include_full_sequence=False, few_shot_count=0),
        replace(base_config, experiment_name=EXPERIMENT_PRESETS["E2"]["experiment_name"], candidate_strategy="full_catalog", label_card_style="short_desc", include_full_sequence=False, few_shot_count=0),
        replace(base_config, experiment_name=EXPERIMENT_PRESETS["E3"]["experiment_name"], candidate_strategy="full_catalog", label_card_style="short_desc", include_full_sequence=True, few_shot_count=0),
    ]
    for top_k in (5, 10, 20):
        configs.append(replace(base_config, experiment_name=EXPERIMENT_PRESETS["E4"]["experiment_name"], candidate_strategy="topk_prototype", label_card_style="short_desc", include_full_sequence=False, few_shot_count=0, top_k=top_k))
        configs.append(replace(base_config, experiment_name=EXPERIMENT_PRESETS["E5"]["experiment_name"], candidate_strategy="topk_prototype", label_card_style="short_desc", include_full_sequence=True, few_shot_count=0, top_k=top_k))
        for few_shot_count in (1, 2):
            configs.append(replace(base_config, experiment_name=EXPERIMENT_PRESETS["E6"]["experiment_name"], candidate_strategy="topk_prototype", label_card_style="short_desc", include_full_sequence=False, few_shot_count=few_shot_count, top_k=top_k))
    return configs


def run_suite(base_config: RunConfig, artifact_root: Path) -> tuple[dict, Path]:
    suite_dir = _resolve_artifact_dir(artifact_root=artifact_root, config=replace(base_config, experiment_name="suite"))
    validation_config = replace(base_config, split="valid")
    validation_runs = []
    best_result = None

    for config in build_suite_configs(validation_config):
        metrics, run_dir = run_single(config, artifact_root=suite_dir / "runs")
        result = {
            "config": config.to_dict(),
            "metrics": metrics,
            "run_dir": str(run_dir),
        }
        validation_runs.append(result)
        if best_result is None or _config_score(metrics) > _config_score(best_result["metrics"]):
            best_result = result

    assert best_result is not None
    selected_config = RunConfig(**best_result["config"])
    selected_test_runs = []
    for dataset_id in _related_dataset_ids(base_config.dataset_id):
        config = replace(selected_config, dataset_id=dataset_id, split="test")
        metrics, run_dir = run_single(config, artifact_root=suite_dir / "selected")
        selected_test_runs.append(
            {
                "config": config.to_dict(),
                "metrics": metrics,
                "run_dir": str(run_dir),
            }
        )

    summary = {
        "validation_runs": validation_runs,
        "selected_config": selected_config.to_dict(),
        "selected_validation_metrics": best_result["metrics"],
        "test_runs": selected_test_runs,
    }
    _write_json(suite_dir / "suite_summary.json", summary)
    return summary, suite_dir


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fragment-level LLM benchmark runner")
    parser.add_argument("--dataset_id", required=True, help="Example: VenusX_Res_Act_MF50")
    parser.add_argument("--experiment", default="E0", choices=sorted(EXPERIMENT_PRESETS))
    parser.add_argument("--split", default="test", choices=["train", "valid", "validation", "test"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--prompt_template", default="fragment_cls_v1")
    parser.add_argument("--label_card_style", choices=["name_only", "short_desc", "rich_desc"], default=None)
    parser.add_argument("--include_full_sequence", action="store_true", default=None)
    parser.add_argument("--no_full_sequence", action="store_false", dest="include_full_sequence")
    parser.add_argument("--few_shot_count", type=int, default=None)
    parser.add_argument("--candidate_strategy", choices=["full_catalog", "topk_prototype"], default=None)
    parser.add_argument("--model_provider", default=parser_defaults()["model_provider"], choices=["mock", "replay", "agent"])
    parser.add_argument("--model_name", default=parser_defaults()["model_name"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--artifact_root", default="artifacts/evaluation_llm")
    parser.add_argument("--suite", action="store_true", help="Run the validation selection suite, then test MF50/MF70/MF90")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    config = build_run_config(args)
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    if args.suite:
        summary, suite_dir = run_suite(config, artifact_root=artifact_root)
        print(json.dumps({"suite_dir": str(suite_dir), "selected_config": summary["selected_config"]}, indent=2))
        return 0

    metrics, run_dir = run_single(config, artifact_root=artifact_root)
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
