"""Data structures for the agent evaluation framework."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvaluationExample:
    """One row from a VenusX CSV file, representing a single labelled fragment."""

    uid: str
    seq_full: str
    interpro_id: str                            # ground-truth InterPro accession
    # Ground-truth fragment position(s) — 1-indexed inclusive (start, end) pairs.
    # Multi-fragment examples have more than one tuple.
    fragment_parts: tuple[tuple[int, int], ...]


@dataclass
class EvalResult:
    """Evaluation outcome for a single example."""

    uid: str
    interpro_id: str
    # Whether the agent returned any annotation with the correct interpro_id.
    label_found: bool

    # Residue-level counts (only meaningful when label_found is True).
    residue_tp: int = 0
    residue_fp: int = 0
    residue_fn: int = 0

    # Fragment-level counts (IoU-based matching, only meaningful when label_found is True).
    fragment_tp: int = 0
    fragment_fp: int = 0
    fragment_fn: int = 0

    # Non-None if the agent raised an exception for this example.
    error: str | None = None

    @property
    def residue_precision(self) -> float | None:
        denom = self.residue_tp + self.residue_fp
        return self.residue_tp / denom if denom else None

    @property
    def residue_recall(self) -> float | None:
        denom = self.residue_tp + self.residue_fn
        return self.residue_tp / denom if denom else None

    @property
    def residue_f1(self) -> float | None:
        p, r = self.residue_precision, self.residue_recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fragment_precision(self) -> float | None:
        denom = self.fragment_tp + self.fragment_fp
        return self.fragment_tp / denom if denom else None

    @property
    def fragment_recall(self) -> float | None:
        denom = self.fragment_tp + self.fragment_fn
        return self.fragment_tp / denom if denom else None

    @property
    def fragment_f1(self) -> float | None:
        p, r = self.fragment_precision, self.fragment_recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if (p + r) else 0.0
