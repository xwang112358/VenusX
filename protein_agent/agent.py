"""ProteinAgent — Claude tool-use loop for InterPro site annotation.

The agent receives a protein sequence, decides to call the `search_interpro`
tool (backed by EBI InterProScan 5), processes the returned annotations, and
returns a structured AgentResult.

Architecture:
    User → ProteinAgent.run(sequence)
                │
                └── Claude (tool-use loop)
                        ├── calls search_interpro(sequence)  ← InterProScanTool
                        └── returns when stop_reason == "end_turn"
"""
from __future__ import annotations

import json
import os

import anthropic

from protein_agent.records import AgentResult, SiteAnnotation
from protein_agent.tools.interpro_scan import InterProScanTool

# ---------------------------------------------------------------------------
# Site types we consider "functional site" annotations
# ---------------------------------------------------------------------------
_SITE_TYPES = {"ACTIVE_SITE", "BINDING_SITE", "CONSERVED_SITE", "PTM"}

# ---------------------------------------------------------------------------
# Tool schema registered with Claude
# ---------------------------------------------------------------------------
_TOOL_SCHEMA: dict = {
    "name": "search_interpro",
    "description": (
        "Submit a protein sequence to EBI InterProScan and return all matching "
        "InterPro annotations, including active sites, binding sites, conserved "
        "sites, domains, and post-translational modifications.  Each annotation "
        "includes the InterPro accession, entry name, functional type, and the "
        "residue positions (1-indexed, inclusive) where it was matched."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sequence": {
                "type": "string",
                "description": (
                    "Amino acid sequence in single-letter code, no spaces or gaps."
                ),
            }
        },
        "required": ["sequence"],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a protein function analyst. "
    "When given a protein sequence, use the search_interpro tool to retrieve "
    "InterPro annotations for that sequence. "
    "Do not fabricate annotations — always use the tool to obtain them. "
    "After receiving the tool results, confirm the annotations have been retrieved."
)


def _build_user_prompt(sequence: str) -> str:
    return (
        f"Identify all InterPro annotations — especially active sites and "
        f"binding sites — for the following protein sequence:\n\n{sequence}"
    )


def _annotations_to_json(annotations: list[SiteAnnotation]) -> str:
    """Serialise a list of SiteAnnotation objects to a JSON string for the tool result."""
    return json.dumps(
        [
            {
                "accession": a.accession,
                "name": a.name,
                "site_type": a.site_type,
                "locations": [{"start": s, "end": e} for s, e in a.locations],
            }
            for a in annotations
        ],
        indent=2,
    )


class ProteinAgent:
    """Claude-backed agent that annotates a protein sequence via InterProScan.

    Parameters
    ----------
    email:
        Valid e-mail address required by EBI InterProScan.
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY`` environment
        variable if *None*.
    model:
        Claude model ID (default ``claude-opus-4-6``).
    max_tokens:
        Maximum tokens for each Claude response in the loop.
    poll_interval:
        Seconds between InterProScan status polls.
    max_wait:
        Maximum seconds to wait for an InterProScan job to finish.
    timeout:
        Per-request HTTP timeout for InterProScan calls.
    """

    def __init__(
        self,
        email: str,
        api_key: str | None = None,
        model: str = "claude-opus-4-6",
        max_tokens: int = 1024,
        poll_interval: int = 15,
        max_wait: int = 300,
        timeout: int = 30,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.tool = InterProScanTool(
            email=email,
            poll_interval=poll_interval,
            max_wait=max_wait,
            timeout=timeout,
        )

    def run(self, sequence: str) -> AgentResult:
        """Run the tool-use loop and return site annotations for *sequence*."""
        sequence = sequence.strip()
        messages: list[dict] = [
            {"role": "user", "content": _build_user_prompt(sequence)}
        ]

        tool_call_count = 0
        all_annotations: list[SiteAnnotation] = []
        final_usage: dict = {}

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                tools=[_TOOL_SCHEMA],
                messages=messages,
                thinking={"type": "adaptive"},
            )

            # Append the full assistant turn (preserves thinking / tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                # Claude is done (end_turn or other terminal state)
                if response.usage:
                    final_usage = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                break

            # Execute every tool call Claude requested in this turn
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_call_count += 1
                seq_input: str = block.input.get("sequence", sequence)
                annotations = self.tool.search(seq_input)
                all_annotations.extend(annotations)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _annotations_to_json(annotations),
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        site_annotations = [a for a in all_annotations if a.site_type in _SITE_TYPES]

        return AgentResult(
            annotations=tuple(all_annotations),
            site_annotations=tuple(site_annotations),
            metadata={
                "model": self.model,
                "tool_calls": tool_call_count,
                "usage": final_usage,
            },
        )
