"""Residue-level and fragment-level evaluation metrics.

Both metric functions work on sets/tuples of integer residue positions, keeping
the logic independent of the protein_agent and evaluation_agent data structures.
"""
from __future__ import annotations

from evaluation_agent.records import EvalResult


# ---------------------------------------------------------------------------
# Per-example metrics
# ---------------------------------------------------------------------------

def residue_metrics(
    true_residues: frozenset[int],
    pred_residues: frozenset[int],
) -> dict[str, int | float | None]:
    """Compute residue-level TP/FP/FN and derived P/R/F1.

    Parameters
    ----------
    true_residues:
        Residue positions covered by the ground-truth annotation.
    pred_residues:
        Residue positions covered by the predicted annotation.

    Returns
    -------
    dict with keys: tp, fp, fn, precision, recall, f1
    """
    tp = len(true_residues & pred_residues)
    fp = len(pred_residues - true_residues)
    fn = len(true_residues - pred_residues)

    precision = tp / (tp + fp) if (tp + fp) else None
    recall    = tp / (tp + fn) if (tp + fn) else None

    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    else:
        f1 = None

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def fragment_metrics(
    true_parts: tuple[tuple[int, int], ...],
    pred_parts: tuple[tuple[int, int], ...],
    iou_threshold: float = 0.5,
) -> dict[str, int | float | None]:
    """Compute fragment-level TP/FP/FN using residue-set IoU matching.

    A predicted fragment is a *true positive* if it overlaps a ground-truth
    fragment with IoU ≥ *iou_threshold* (measured as residue set overlap).
    Each ground-truth fragment can be matched at most once.

    Parameters
    ----------
    true_parts:
        Ground-truth (start, end) ranges — 1-indexed, inclusive.
    pred_parts:
        Predicted (start, end) ranges — 1-indexed, inclusive.
    iou_threshold:
        Minimum IoU for a match (default 0.5).

    Returns
    -------
    dict with keys: tp, fp, fn, precision, recall, f1
    """
    matched_true: set[int] = set()   # indices into true_parts
    tp = 0
    fp = 0

    for pred_start, pred_end in pred_parts:
        pred_set = frozenset(range(pred_start, pred_end + 1))
        best_iou = 0.0
        best_idx = -1

        for idx, (true_start, true_end) in enumerate(true_parts):
            if idx in matched_true:
                continue
            true_set = frozenset(range(true_start, true_end + 1))
            intersection = len(pred_set & true_set)
            union = len(pred_set | true_set)
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx >= 0:
            tp += 1
            matched_true.add(best_idx)
        else:
            fp += 1

    fn = len(true_parts) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) else None
    recall    = tp / (tp + fn) if (tp + fn) else None

    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    else:
        f1 = None

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(results: list[EvalResult]) -> dict:
    """Compute dataset-level summary statistics from a list of EvalResult objects.

    Returns
    -------
    dict with keys:
        n_total, n_errors, label_recall,
        mean_residue_{precision,recall,f1},
        mean_fragment_{precision,recall,f1}
    """
    n_total = len(results)
    n_errors = sum(1 for r in results if r.error is not None)
    n_label_found = sum(1 for r in results if r.label_found)

    label_recall = n_label_found / n_total if n_total else 0.0

    # Only average over non-error, label-found examples for position metrics
    valid = [r for r in results if r.error is None and r.label_found]

    def _mean(values: list[float | None]) -> float | None:
        finite = [v for v in values if v is not None]
        return sum(finite) / len(finite) if finite else None

    return {
        "n_total": n_total,
        "n_errors": n_errors,
        "n_label_found": n_label_found,
        "label_recall": label_recall,
        "mean_residue_precision": _mean([r.residue_precision for r in valid]),
        "mean_residue_recall":    _mean([r.residue_recall    for r in valid]),
        "mean_residue_f1":        _mean([r.residue_f1        for r in valid]),
        "mean_fragment_precision": _mean([r.fragment_precision for r in valid]),
        "mean_fragment_recall":    _mean([r.fragment_recall    for r in valid]),
        "mean_fragment_f1":        _mean([r.fragment_f1        for r in valid]),
    }
