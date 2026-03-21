from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from env_utils import load_default_env_file
from evaluation_llm.label_catalog import LabelCatalog
from evaluation_llm.records import FragmentExample, LabelCard, ModelResponse


def _label_text_score(fragment_text: str, label_text: str) -> tuple[int, int]:
    if len(fragment_text) < 3:
        fragment_tokens = {fragment_text.upper()} if fragment_text else set()
    else:
        fragment_tokens = {fragment_text[index : index + 3] for index in range(len(fragment_text) - 2)}
    label_tokens = {token.lower() for token in label_text.replace(",", " ").split()}
    overlap = sum(1 for token in label_tokens if token and token.upper() in fragment_tokens)
    return overlap, -len(label_text)


class MockModelBackend:
    def __init__(self, catalog: LabelCatalog, mode: str = "oracle") -> None:
        self.catalog = catalog
        self.mode = mode

    def generate(
        self,
        prompt: str,
        example: FragmentExample,
        label_cards: tuple[LabelCard, ...],
    ) -> ModelResponse:
        if self.mode == "oracle":
            top_ids = [example.interpro_id]
            metadata = {"mode": "oracle"}
        elif self.mode in {"first_label", "first_candidate"}:
            top_ids = [label_cards[0].accession] if label_cards else []
            metadata = {"mode": "first_label"}
        else:
            fragment_text = example.compact_fragment()
            ranked = sorted(
                label_cards,
                key=lambda card: _label_text_score(fragment_text, f"{card.name} {card.short_desc}"),
                reverse=True,
            )
            top_ids = [ranked[0].accession] if ranked else []
            metadata = {"mode": "heuristic"}

        payload = {
            "top_ids": top_ids,
            "reasoning_summary": (
                "Oracle mock backend returns the gold label."
                if self.mode == "oracle"
                else "Mock backend ranks labels using simple deterministic heuristics."
            ),
            "abstain": not top_ids,
        }
        return ModelResponse(raw_text=json.dumps(payload), metadata=metadata)


class ReplayModelBackend:
    def __init__(self, replay_path: str | Path) -> None:
        self.responses: dict[str, str] = {}
        replay_path = Path(replay_path)
        for line in replay_path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            uid = payload["uid"]
            raw_text = payload["raw_text"]
            self.responses[uid] = raw_text

    def generate(
        self,
        prompt: str,
        example: FragmentExample,
        label_cards: tuple[LabelCard, ...],
    ) -> ModelResponse:
        if example.uid not in self.responses:
            raise KeyError(f"No replay response for uid={example.uid}")
        return ModelResponse(raw_text=self.responses[example.uid], metadata={"mode": "replay"})


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


class OpenRouterModelBackend:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        api_key: str | None = None,
        api_url: str | None = None,
        app_url: str | None = None,
        app_title: str | None = None,
        max_tokens: int = 256,
        timeout_seconds: float = 120.0,
    ) -> None:
        load_default_env_file(env_override_var="OPENROUTER_ENV_FILE")
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.api_url = api_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
        self.app_url = app_url or os.environ.get("OPENROUTER_HTTP_REFERER")
        self.app_title = app_title or os.environ.get("OPENROUTER_TITLE")
        self.max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", str(max_tokens)))
        self.timeout_seconds = float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", str(timeout_seconds)))

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for model_provider='openrouter'. "
                "Set it in the environment before running the benchmark."
            )

    def generate(
        self,
        prompt: str,
        example: FragmentExample,
        label_cards: tuple[LabelCard, ...],
    ) -> ModelResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "user": example.uid,
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )

        if response.status_code >= 400:
            body_preview = response.text[:1000]
            raise RuntimeError(
                f"OpenRouter request failed with status {response.status_code}: {body_preview}"
            )

        response_json = response.json()
        choices = response_json.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter response did not include choices: {response_json}")

        message = choices[0].get("message") or {}
        raw_text = _content_to_text(message.get("content"))
        if not raw_text:
            raise RuntimeError(f"OpenRouter response did not include text content: {response_json}")

        metadata = {
            "mode": "openrouter",
            "response_id": response_json.get("id"),
            "response_model": response_json.get("model"),
            "usage": response_json.get("usage"),
        }
        return ModelResponse(raw_text=raw_text, metadata=metadata)


class PlaceholderAgentBackend:
    def generate(
        self,
        prompt: str,
        example: FragmentExample,
        label_cards: tuple[LabelCard, ...],
    ) -> ModelResponse:
        raise NotImplementedError(
            "Agent evaluation is reserved for a later phase. "
            "Implement a real agent backend to use this provider."
        )


class InterProAgentBackend:
    """Thin adapter that wraps ProteinAgent for use inside the evaluation_llm benchmark.

    Translates AgentResult → ModelResponse so that the existing parse_model_response()
    pipeline can score agent predictions alongside LLM baselines.

    Environment variables
    ---------------------
    ANTHROPIC_API_KEY     — passed to ProteinAgent / Claude
    INTERPROSCAN_EMAIL    — required by EBI terms of service
    """

    def __init__(self, model_name: str = "claude-opus-4-6") -> None:
        load_default_env_file(env_override_var="OPENROUTER_ENV_FILE")
        from protein_agent.agent import ProteinAgent

        email = os.environ.get("INTERPROSCAN_EMAIL", "")
        if not email:
            raise ValueError(
                "INTERPROSCAN_EMAIL is required for model_provider='interpro_agent'. "
                "Set it in your environment or .env file."
            )
        self._agent = ProteinAgent(email=email, model=model_name)

    def generate(
        self,
        prompt: str,
        example: FragmentExample,
        label_cards: tuple[LabelCard, ...],
    ) -> ModelResponse:
        sequence = example.seq_full or example.compact_fragment()
        agent_result = self._agent.run(sequence)

        # Map site annotations to catalog accessions (ranked by site type priority)
        site_type_rank = {"ACTIVE_SITE": 0, "BINDING_SITE": 1, "CONSERVED_SITE": 2}
        ranked = sorted(
            agent_result.site_annotations,
            key=lambda a: site_type_rank.get(a.site_type, 99),
        )
        top_ids = [a.accession for a in ranked][:3]

        payload = {
            "top_ids": top_ids,
            "reasoning_summary": (
                f"InterProScan returned {len(agent_result.annotations)} annotation(s); "
                f"{len(agent_result.site_annotations)} functional site(s) identified."
            ),
            "abstain": len(top_ids) == 0,
        }
        return ModelResponse(
            raw_text=json.dumps(payload),
            metadata=agent_result.metadata,
        )


def create_model_backend(
    model_provider: str,
    model_name: str,
    catalog: LabelCatalog,
    temperature: float = 0.0,
):
    if model_provider == "mock":
        return MockModelBackend(catalog=catalog, mode=model_name)
    if model_provider == "replay":
        return ReplayModelBackend(model_name)
    if model_provider == "openrouter":
        return OpenRouterModelBackend(model_name=model_name, temperature=temperature)
    if model_provider == "agent":
        return PlaceholderAgentBackend()
    if model_provider == "interpro_agent":
        return InterProAgentBackend(model_name=model_name)
    raise ValueError(f"Unsupported model_provider={model_provider!r}")
