"""ProteinAgent — OpenAI tool-use loop for InterPro site annotation.

The agent receives a protein sequence, decides to call the `search_interpro`
tool (backed by EBI InterProScan 5), processes the returned annotations, and
returns a structured AgentResult.

Architecture:
    User → ProteinAgent.run(sequence)
                │
                └── LLM (tool-use loop via OpenAI chat completions)
                        ├── calls search_interpro(sequence)  ← InterProScanTool
                        └── returns when finish_reason == "stop"

Supports both Azure OpenAI (default) and plain OpenAI endpoints — the only
difference is which client class is instantiated in __init__.
"""
from __future__ import annotations

import json
import os

from protein_agent.records import AgentResult, SiteAnnotation
from protein_agent.tools.interpro_scan import InterProScanTool

# ---------------------------------------------------------------------------
# Site types we consider "functional site" annotations
# ---------------------------------------------------------------------------
_SITE_TYPES = {"ACTIVE_SITE", "BINDING_SITE", "CONSERVED_SITE", "PTM"}

# ---------------------------------------------------------------------------
# Tool schema — OpenAI function-calling format
# ---------------------------------------------------------------------------
_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "search_interpro",
        "description": (
            "Submit a protein sequence to EBI InterProScan and return all matching "
            "InterPro annotations, including active sites, binding sites, conserved "
            "sites, domains, and post-translational modifications.  Each annotation "
            "includes the InterPro accession, entry name, functional type, and the "
            "residue positions (1-indexed, inclusive) where it was matched."
        ),
        "parameters": {
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
    },
}

# ---------------------------------------------------------------------------
# Prompt helpers
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
    """OpenAI-backed agent that annotates a protein sequence via InterProScan.

    Supports both Azure OpenAI (``use_azure=True``, the default) and plain
    OpenAI (``use_azure=False``).  The only difference is the client
    constructor; all downstream logic is identical.

    Parameters
    ----------
    email:
        Valid e-mail address required by EBI InterProScan.
    model:
        Model / deployment name.  For Azure this is the deployment name
        (e.g. ``"gpt-4o"``); for plain OpenAI it is the model ID.
    max_tokens:
        Maximum tokens for each LLM response in the loop.
    use_azure:
        If *True* (default), construct an ``AzureOpenAI`` client using
        ``AZURE_OPENAI_ENDPOINT`` and ``AZURE_OPENAI_API_KEY`` from the
        environment.  If *False*, construct an ``OpenAI`` client using
        ``OPENAI_API_KEY``.
    poll_interval:
        Seconds between InterProScan status polls.
    max_wait:
        Maximum seconds to wait for an InterProScan job to finish.
    timeout:
        Per-request HTTP timeout for InterProScan calls.
    _client:
        Optional pre-constructed OpenAI client — used in unit tests to
        inject a mock without patching module imports.
    """

    def __init__(
        self,
        email: str,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        use_azure: bool = True,
        poll_interval: int = 15,
        max_wait: int = 300,
        timeout: int = 30,
        _client=None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens

        if _client is not None:
            self.client = _client
        elif use_azure:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version="2024-02-01",
            )
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(sequence)},
        ]

        tool_call_count = 0
        all_annotations: list[SiteAnnotation] = []
        final_usage: dict = {}

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                tools=[_TOOL_SCHEMA],
                messages=messages,
            )

            choice = response.choices[0]
            msg = choice.message

            # Append the assistant turn (preserves tool_calls metadata)
            messages.append(msg)

            if choice.finish_reason != "tool_calls":
                # LLM is done (stop, length, or other terminal state)
                if response.usage:
                    final_usage = response.usage.model_dump()
                break

            # Execute every tool call the LLM requested in this turn
            for tc in msg.tool_calls or []:
                tool_call_count += 1
                args = json.loads(tc.function.arguments)
                seq_input: str = args.get("sequence", sequence)
                annotations = self.tool.search(seq_input)
                all_annotations.extend(annotations)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _annotations_to_json(annotations),
                    }
                )

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
