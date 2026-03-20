#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MODELS = [
    "deepseek/deepseek-chat-v3.1",
    "openai/gpt-5",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def short_model_name(model_name: str | None) -> str:
    if not model_name:
        return "unknown"
    return model_name.split("/")[-1]


def compact_run_label(dataset_id: str | None, experiment_name: str | None) -> str:
    dataset_label = (dataset_id or "unknown_dataset").replace("VenusX_Res_", "").replace("_MF50", "")
    experiment_label = (experiment_name or "unknown_experiment").split("_")[0]
    return f"{dataset_label} {experiment_label}"


def discover_runs(artifact_root: Path, target_models: set[str] | None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []

    for config_path in sorted(artifact_root.glob("*/*/resolved_config.json")):
        config = load_json(config_path)
        settings = config.get("settings", {})
        dataset_info = config.get("dataset_info", {})
        alignment = config.get("alignment_summary", {})
        run_dir = config_path.parent

        model_name = settings.get("model_name")
        if target_models is not None and model_name not in target_models:
            continue
        if not (run_dir / "metrics.json").exists():
            continue
        if not (run_dir / "records.jsonl").exists():
            continue

        runs.append(
            {
                "run_dir": run_dir,
                "dataset_id": dataset_info.get("dataset_id"),
                "track_name": dataset_info.get("track_name"),
                "catalog_path": dataset_info.get("catalog_path"),
                "experiment_name": settings.get("experiment_name"),
                "label_card_style": settings.get("label_card_style"),
                "model_name": model_name,
                "model_short": short_model_name(model_name),
                "example_count": alignment.get("example_count"),
                "run_label": (
                    f"{short_model_name(model_name)} "
                    f"{compact_run_label(dataset_info.get('dataset_id'), settings.get('experiment_name'))}"
                ),
            }
        )

    runs.sort(key=lambda row: (row["dataset_id"] or "", row["experiment_name"] or "", row["model_name"] or ""))
    return runs


def build_catalog_lookup(runs: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    seen_catalogs: set[tuple[str, str]] = set()

    for run in runs:
        dataset_id = run["dataset_id"]
        catalog_path = run["catalog_path"]
        if not dataset_id or not catalog_path:
            continue
        key = (dataset_id, catalog_path)
        if key in seen_catalogs:
            continue
        seen_catalogs.add(key)

        for item in load_json(Path(catalog_path)):
            metadata = item.get("metadata", {})
            accession = metadata.get("accession")
            if accession:
                lookup[(dataset_id, accession)] = metadata.get("name", "")

    return lookup


def label_name(
    dataset_id: str | None,
    accession: str | None,
    lookup: dict[tuple[str, str], str],
) -> str | None:
    if not dataset_id or not accession:
        return None
    return lookup.get((dataset_id, accession))


def failure_type(
    truth_id: str | None,
    predicted_top_id: str | None,
    parse_success: bool | None,
    abstain: bool | None,
    backend_error: str | None,
) -> str | None:
    if backend_error:
        return "backend_error"
    if parse_success is False:
        return "parse_failure"
    if abstain:
        return "abstain"
    if predicted_top_id != truth_id:
        return "wrong_label"
    return None


def build_metrics_row(run: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = load_json(Path(run["run_dir"]) / "metrics.json")
    main = metrics.get("main_paper_table", {})
    llm = metrics.get("supplemental_llm_table", {})

    wrong_label_count = 0
    backend_error_count = 0
    parse_failure_count = 0
    abstain_count = 0

    for item in records:
        example = item.get("example", {})
        prediction = item.get("prediction", {})
        response_metadata = item.get("response_metadata", {})
        if not isinstance(response_metadata, dict):
            response_metadata = {}

        truth_id = example.get("interpro_id")
        predicted_top_id = item.get("predicted_top_id")
        parse_success = prediction.get("parse_success")
        abstain = prediction.get("abstain")
        backend_error = response_metadata.get("backend_error")

        if backend_error:
            backend_error_count += 1
        if parse_success is False:
            parse_failure_count += 1
        if abstain:
            abstain_count += 1
        if predicted_top_id is not None and predicted_top_id != truth_id:
            wrong_label_count += 1

    return {
        "run_label": run["run_label"],
        "dataset_id": run["dataset_id"],
        "track_name": run["track_name"],
        "experiment_name": run["experiment_name"],
        "label_card_style": run["label_card_style"],
        "model_name": run["model_name"],
        "model_short": run["model_short"],
        "count": main.get("count", llm.get("count")),
        "accuracy": main.get("accuracy"),
        "macro_f1": main.get("macro_f1"),
        "mcc": main.get("mcc"),
        "top3_acc": llm.get("top3_acc"),
        "parse_success_rate": llm.get("parse_success_rate"),
        "coverage": llm.get("coverage"),
        "wrong_label_count": wrong_label_count,
        "parse_failure_count": parse_failure_count,
        "backend_error_count": backend_error_count,
        "abstain_count": abstain_count,
    }


def build_representative_failures(
    run: dict[str, Any],
    records: list[dict[str, Any]],
    lookup: dict[tuple[str, str], str],
    failures_per_run: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for item in records:
        example = item.get("example", {})
        prediction = item.get("prediction", {})
        response_metadata = item.get("response_metadata", {})
        if not isinstance(response_metadata, dict):
            response_metadata = {}

        truth_id = example.get("interpro_id")
        predicted_top_id = item.get("predicted_top_id")
        parse_success = prediction.get("parse_success")
        abstain = prediction.get("abstain")
        backend_error = response_metadata.get("backend_error")
        current_failure_type = failure_type(
            truth_id=truth_id,
            predicted_top_id=predicted_top_id,
            parse_success=parse_success,
            abstain=abstain,
            backend_error=backend_error,
        )
        if current_failure_type is None:
            continue

        top_ids = prediction.get("top_ids") or []
        rows.append(
            {
                "run_label": run["run_label"],
                "dataset_id": run["dataset_id"],
                "track_name": run["track_name"],
                "experiment_name": run["experiment_name"],
                "label_card_style": run["label_card_style"],
                "model_name": run["model_name"],
                "model_short": run["model_short"],
                "failure_type": current_failure_type,
                "uid": example.get("uid"),
                "fragment_length": example.get("fragment_length"),
                "fragment": example.get("seq_fragment_raw"),
                "truth_id": truth_id,
                "truth_name": label_name(run["dataset_id"], truth_id, lookup),
                "predicted_top_id": predicted_top_id,
                "predicted_name": label_name(run["dataset_id"], predicted_top_id, lookup),
                "top_ids": ", ".join(top_ids),
                "reasoning_summary": prediction.get("reasoning_summary"),
                "parse_error": prediction.get("parse_error"),
                "backend_error": backend_error,
            }
        )

    if not rows:
        return []

    failures = pd.DataFrame(rows)
    failures["confusion_count"] = failures.groupby(
        ["truth_id", "predicted_top_id"],
        dropna=False,
    )["uid"].transform("size")
    failure_priority = {
        "wrong_label": 0,
        "abstain": 1,
        "parse_failure": 2,
        "backend_error": 3,
    }
    failures["failure_priority"] = failures["failure_type"].map(failure_priority).fillna(99)
    failures = failures.sort_values(
        ["failure_priority", "confusion_count", "fragment_length", "uid"],
        ascending=[True, False, True, True],
    )
    failures = failures.head(failures_per_run).drop(columns=["failure_priority", "confusion_count"])
    return failures.to_dict(orient="records")


def format_rate(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{100.0 * float(value):.1f}%"


def format_decimal(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def write_summary(output_path: Path, metrics_df: pd.DataFrame, failures_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    lines.append(f"Completed runs: {len(metrics_df)}")
    if not metrics_df.empty:
        model_list = ", ".join(sorted(metrics_df["model_name"].dropna().unique()))
        lines.append(f"Models included: {model_list}")

    if not metrics_df.empty:
        best_row = metrics_df.loc[metrics_df["accuracy"].idxmax()]
        worst_row = metrics_df.loc[metrics_df["accuracy"].idxmin()]
        lines.append(f"Best accuracy: {best_row['run_label']} ({format_rate(best_row['accuracy'])})")
        lines.append(f"Lowest accuracy: {worst_row['run_label']} ({format_rate(worst_row['accuracy'])})")

    if not failures_df.empty:
        lines.append(f"Representative failures saved: {len(failures_df)}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def create_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Summarize completed DeepSeek runs with metrics and representative failures."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=repo_root / "artifacts" / "evaluation_llm",
        help="Root directory containing evaluation_llm artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "analysis" / "deepseek_results",
        help="Directory for the output CSVs.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model names to analyze. If omitted explicitly, all completed models are included.",
    )
    parser.add_argument(
        "--failures-per-run",
        type=int,
        default=5,
        help="How many representative failures to keep per run.",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_models = set(args.models) if args.models else None
    runs = discover_runs(args.artifact_root.resolve(), target_models)
    if not runs:
        if target_models is None:
            raise SystemExit("No completed runs found.")
        requested_models = ", ".join(sorted(target_models))
        raise SystemExit(f"No completed runs found for models: {requested_models}")

    lookup = build_catalog_lookup(runs)

    metrics_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for run in runs:
        records = load_jsonl(Path(run["run_dir"]) / "records.jsonl")
        metrics_rows.append(build_metrics_row(run, records))
        failure_rows.extend(
            build_representative_failures(
                run=run,
                records=records,
                lookup=lookup,
                failures_per_run=args.failures_per_run,
            )
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["dataset_id", "experiment_name", "model_name"]
    ).reset_index(drop=True)
    failures_df = pd.DataFrame(failure_rows).sort_values(
        ["dataset_id", "experiment_name", "model_name", "failure_type", "uid"]
    ).reset_index(drop=True)

    metrics_path = output_dir / "deepseek_metrics.csv"
    failures_path = output_dir / "representative_failures.csv"
    summary_path = output_dir / "summary.txt"

    metrics_df.to_csv(metrics_path, index=False)
    failures_df.to_csv(failures_path, index=False)
    summary_lines = write_summary(summary_path, metrics_df, failures_df)

    print("\n".join(summary_lines))
    print()
    print(f"Metrics: {metrics_path}")
    print(f"Representative failures: {failures_path}")
    print(f"Summary: {summary_path}")
    print()

    metrics_view = metrics_df[
        [
            "run_label",
            "model_short",
            "accuracy",
            "macro_f1",
            "mcc",
            "top3_acc",
            "parse_success_rate",
            "coverage",
            "wrong_label_count",
            "parse_failure_count",
            "backend_error_count",
        ]
    ].copy()
    for column in ("accuracy", "macro_f1", "top3_acc", "parse_success_rate", "coverage"):
        metrics_view[column] = metrics_view[column].map(format_rate)
    metrics_view["mcc"] = metrics_view["mcc"].map(format_decimal)
    print(metrics_view.to_string(index=False))


if __name__ == "__main__":
    main()
