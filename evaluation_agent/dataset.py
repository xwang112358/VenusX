"""Load VenusX CSV files into EvaluationExample objects.

Expected CSV columns (same schema as evaluation_llm):
    uid, seq_full, interpro_id, start, end
    [start / end are pipe-delimited for multi-fragment examples]
"""
from __future__ import annotations

import csv
from pathlib import Path

from evaluation_agent.records import EvaluationExample


def load_examples(csv_path: Path | str) -> list[EvaluationExample]:
    """Read *csv_path* and return one :class:`EvaluationExample` per row.

    Parameters
    ----------
    csv_path:
        Path to a VenusX split CSV (train / valid / test).

    Raises
    ------
    ValueError
        If a required column is missing or a row has malformed position data.
    """
    csv_path = Path(csv_path)
    examples: list[EvaluationExample] = []

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _check_columns(reader.fieldnames or [], csv_path)

        for lineno, row in enumerate(reader, start=2):  # 1-indexed; row 1 is header
            uid = row["uid"].strip()
            seq_full = row["seq_full"].strip()
            interpro_id = row["interpro_id"].strip()

            starts_raw = row["start"].strip()
            ends_raw = row["end"].strip()

            try:
                fragment_parts = _parse_positions(starts_raw, ends_raw)
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{lineno} — bad position data for uid={uid!r}: {exc}"
                ) from exc

            examples.append(
                EvaluationExample(
                    uid=uid,
                    seq_full=seq_full,
                    interpro_id=interpro_id,
                    fragment_parts=fragment_parts,
                )
            )

    return examples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = {"uid", "seq_full", "interpro_id", "start", "end"}


def _check_columns(fieldnames: list[str], path: Path) -> None:
    missing = _REQUIRED_COLUMNS - set(fieldnames)
    if missing:
        raise ValueError(
            f"{path}: missing required CSV column(s): {', '.join(sorted(missing))}"
        )


def _parse_positions(
    starts_raw: str, ends_raw: str
) -> tuple[tuple[int, int], ...]:
    """Parse pipe-delimited start/end strings into a tuple of (start, end) pairs."""
    start_parts = starts_raw.split("|")
    end_parts = ends_raw.split("|")

    if len(start_parts) != len(end_parts):
        raise ValueError(
            f"start has {len(start_parts)} parts but end has {len(end_parts)}"
        )

    pairs: list[tuple[int, int]] = []
    for s_str, e_str in zip(start_parts, end_parts):
        s, e = int(s_str.strip()), int(e_str.strip())
        if s > e:
            raise ValueError(f"start {s} > end {e}")
        pairs.append((s, e))

    return tuple(pairs)
