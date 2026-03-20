"""Data structures for the protein agent."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SiteAnnotation:
    """A single InterPro site annotation returned by InterProScan."""

    accession: str          # e.g. "IPR001270"
    name: str               # Human-readable name
    site_type: str          # "ACTIVE_SITE", "BINDING_SITE", "CONSERVED_SITE", etc.
    # Each tuple is a (start, end) residue range — 1-indexed, inclusive.
    locations: tuple[tuple[int, int], ...]

    def residue_set(self) -> frozenset[int]:
        """Return the set of all residue positions covered by this annotation."""
        residues: set[int] = set()
        for start, end in self.locations:
            residues.update(range(start, end + 1))
        return frozenset(residues)


@dataclass(frozen=True)
class AgentResult:
    """Output produced by ProteinAgent.run()."""

    annotations: tuple[SiteAnnotation, ...]
    # Subset restricted to active/binding sites (ACTIVE_SITE, BINDING_SITE).
    site_annotations: tuple[SiteAnnotation, ...]
    metadata: dict = field(default_factory=dict)

    def find(self, accession: str) -> SiteAnnotation | None:
        """Return the first annotation matching *accession*, or None."""
        for ann in self.annotations:
            if ann.accession == accession:
                return ann
        return None
