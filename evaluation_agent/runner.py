"""Main evaluation loop for the protein agent."""
from __future__ import annotations

import traceback
from pathlib import Path

from protein_agent.agent import ProteinAgent
from evaluation_agent.dataset import load_examples
from evaluation_agent.metrics import fragment_metrics, residue_metrics
from evaluation_agent.records import EvalResult, EvaluationExample


def run_evaluation(
    csv_path: Path | str,
    agent: ProteinAgent,
    max_examples: int | None = None,
    iou_threshold: float = 0.5,
) -> list[EvalResult]:
    """Run *agent* on every example in *csv_path* and return EvalResult objects.

    Parameters
    ----------
    csv_path:
        Path to a VenusX split CSV.
    agent:
        A configured :class:`~protein_agent.agent.ProteinAgent` instance.
    max_examples:
        Cap on examples to evaluate (useful for smoke tests).
    iou_threshold:
        IoU threshold for fragment-level matching (default 0.5).

    Returns
    -------
    List of :class:`~evaluation_agent.records.EvalResult`, one per example
    (errors are captured rather than re-raised so the run always completes).
    """
    examples = load_examples(csv_path)
    if max_examples is not None:
        examples = examples[:max_examples]

    results: list[EvalResult] = []
    for i, example in enumerate(examples, start=1):
        print(f"[{i}/{len(examples)}] uid={example.uid} interpro={example.interpro_id}", flush=True)
        result = _evaluate_one(example, agent, iou_threshold)
        results.append(result)
        _print_result(result)

    return results


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

def _evaluate_one(
    example: EvaluationExample,
    agent: ProteinAgent,
    iou_threshold: float,
) -> EvalResult:
    try:
        agent_result = agent.run(example.seq_full)
    except Exception:
        return EvalResult(
            uid=example.uid,
            interpro_id=example.interpro_id,
            label_found=False,
            error=traceback.format_exc(),
        )

    # Check whether the correct accession was returned at all
    matching_annotation = agent_result.find(example.interpro_id)
    label_found = matching_annotation is not None

    if not label_found:
        return EvalResult(
            uid=example.uid,
            interpro_id=example.interpro_id,
            label_found=False,
        )

    # Compute residue-level metrics
    true_residues = _parts_to_residues(example.fragment_parts)
    pred_residues = matching_annotation.residue_set()
    res = residue_metrics(true_residues, pred_residues)

    # Compute fragment-level metrics
    pred_parts = matching_annotation.locations
    frag = fragment_metrics(example.fragment_parts, pred_parts, iou_threshold)

    return EvalResult(
        uid=example.uid,
        interpro_id=example.interpro_id,
        label_found=True,
        residue_tp=res["tp"],
        residue_fp=res["fp"],
        residue_fn=res["fn"],
        fragment_tp=frag["tp"],
        fragment_fp=frag["fp"],
        fragment_fn=frag["fn"],
    )


def _parts_to_residues(parts: tuple[tuple[int, int], ...]) -> frozenset[int]:
    residues: set[int] = set()
    for start, end in parts:
        residues.update(range(start, end + 1))
    return frozenset(residues)


def _print_result(result: EvalResult) -> None:
    if result.error:
        print(f"  ERROR: {result.error.splitlines()[-1]}")
    elif result.label_found:
        print(
            f"  label=FOUND  res_f1={result.residue_f1:.3f}  frag_f1={result.fragment_f1:.3f}"
        )
    else:
        print("  label=MISSING")
