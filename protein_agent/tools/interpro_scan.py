"""InterProScan 5 REST API client.

EBI job-dispatcher endpoints used:
  POST   https://www.ebi.ac.uk/Tools/services/rest/iprscan5/run
  GET    https://www.ebi.ac.uk/Tools/services/rest/iprscan5/status/{jobId}
  GET    https://www.ebi.ac.uk/Tools/services/rest/iprscan5/result/{jobId}/json
"""
from __future__ import annotations

import time

import requests

from protein_agent.records import SiteAnnotation

_BASE = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

# EBI considers any entry type containing these strings a "site" annotation.
_SITE_TYPES = {"ACTIVE_SITE", "BINDING_SITE", "CONSERVED_SITE", "PTM"}


class InterProScanError(RuntimeError):
    """Raised when the InterProScan job fails or times out."""


class InterProScanTool:
    """Submit a protein sequence to InterProScan and return site annotations.

    Parameters
    ----------
    email:
        A valid e-mail address — required by EBI's terms of service.
    poll_interval:
        Seconds between status-check requests (default 15).
    max_wait:
        Maximum seconds to wait for the job to finish (default 300).
    timeout:
        Per-request HTTP timeout in seconds (default 30).
    """

    def __init__(
        self,
        email: str,
        poll_interval: int = 15,
        max_wait: int = 300,
        timeout: int = 30,
    ) -> None:
        self.email = email
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, sequence: str) -> list[SiteAnnotation]:
        """Run InterProScan on *sequence* and return all annotated sites.

        Returns a list of :class:`SiteAnnotation` objects, one per distinct
        (accession, site_type) combination found.  Each annotation carries all
        location ranges where that entry was matched.
        """
        sequence = sequence.strip()
        job_id = self._submit(sequence)
        self._poll(job_id)
        return self._parse(job_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _submit(self, sequence: str) -> str:
        url = f"{_BASE}/run"
        data = {
            "email": self.email,
            "sequence": sequence,
            "stype": "protein",
            "goterms": "true",
            "pathways": "false",
        }
        resp = requests.post(url, data=data, timeout=self.timeout)
        if resp.status_code != 200:
            raise InterProScanError(
                f"InterProScan submission failed [{resp.status_code}]: {resp.text[:500]}"
            )
        job_id = resp.text.strip()
        if not job_id:
            raise InterProScanError("InterProScan returned an empty job ID.")
        return job_id

    def _poll(self, job_id: str) -> None:
        url = f"{_BASE}/status/{job_id}"
        deadline = time.monotonic() + self.max_wait
        while time.monotonic() < deadline:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                raise InterProScanError(
                    f"Status check failed [{resp.status_code}]: {resp.text[:200]}"
                )
            status = resp.text.strip().upper()
            if status == "FINISHED":
                return
            if status in {"ERROR", "FAILURE", "NOT_FOUND"}:
                raise InterProScanError(f"InterProScan job {job_id} ended with status: {status}")
            # QUEUED or RUNNING — wait and retry
            time.sleep(self.poll_interval)
        raise InterProScanError(
            f"InterProScan job {job_id} did not finish within {self.max_wait}s."
        )

    def _parse(self, job_id: str) -> list[SiteAnnotation]:
        url = f"{_BASE}/result/{job_id}/json"
        resp = requests.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            raise InterProScanError(
                f"Result fetch failed [{resp.status_code}]: {resp.text[:200]}"
            )
        return _parse_result_json(resp.json())


# ---------------------------------------------------------------------------
# JSON parsing — kept as a pure function so it can be tested without HTTP
# ---------------------------------------------------------------------------

def _parse_result_json(data: dict) -> list[SiteAnnotation]:
    """Extract :class:`SiteAnnotation` objects from an InterProScan JSON result."""
    # Accumulate locations per (accession, name, site_type) key.
    accumulated: dict[tuple[str, str, str], list[tuple[int, int]]] = {}

    results = data.get("results") or []
    for result in results:
        for match in result.get("matches") or []:
            sig = match.get("signature") or {}
            entry = sig.get("entry") or {}

            accession: str = entry.get("accession", "")
            if not accession.startswith("IPR"):
                continue

            name: str = entry.get("name", "")
            site_type: str = (entry.get("type") or "").upper()

            key = (accession, name, site_type)
            locs = accumulated.setdefault(key, [])

            for loc in match.get("locations") or []:
                start = loc.get("start")
                end = loc.get("end")
                if start is not None and end is not None:
                    locs.append((int(start), int(end)))

    annotations: list[SiteAnnotation] = []
    for (accession, name, site_type), locs in accumulated.items():
        if locs:
            annotations.append(
                SiteAnnotation(
                    accession=accession,
                    name=name,
                    site_type=site_type,
                    locations=tuple(sorted(set(locs))),
                )
            )

    return annotations
