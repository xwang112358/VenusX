from __future__ import annotations

import json
from pathlib import Path

from evaluation_llm.catalog import LabelCatalog
from evaluation_llm.interfaces import AgentAdapter, ModelAdapter
from evaluation_llm.types import ModelResponse, PromptContext, PromptPackage


def _label_text_score(fragment_text: str, label_text: str) -> tuple[int, int]:
    if len(fragment_text) < 3:
        fragment_tokens = {fragment_text.upper()} if fragment_text else set()
    else:
        fragment_tokens = {fragment_text[index : index + 3] for index in range(len(fragment_text) - 2)}
    label_tokens = {token.lower() for token in label_text.replace(",", " ").split()}
    overlap = sum(1 for token in label_tokens if token and token.upper() in fragment_tokens)
    return overlap, -len(label_text)


class MockModelAdapter(ModelAdapter):
    def __init__(self, catalog: LabelCatalog, mode: str = "oracle") -> None:
        self.catalog = catalog
        self.mode = mode

    def generate(self, package: PromptPackage, context: PromptContext) -> ModelResponse:
        if self.mode == "oracle":
            top_ids = [context.example.interpro_id]
            metadata = {"mode": "oracle"}
        elif self.mode == "first_candidate":
            top_ids = [context.candidate_records[0].accession] if context.candidate_records else []
            metadata = {"mode": "first_candidate"}
        else:
            fragment_text = context.example.compact_fragment()
            ranked = sorted(
                context.candidate_cards,
                key=lambda card: _label_text_score(fragment_text, f"{card.name} {card.short_desc}"),
                reverse=True,
            )
            top_ids = [ranked[0].accession] if ranked else []
            metadata = {"mode": "heuristic"}

        payload = {
            "top_ids": top_ids,
            "confidence": 1.0 if top_ids else 0.0,
            "abstain": not top_ids,
        }
        return ModelResponse(raw_text=json.dumps(payload), metadata=metadata)


class ReplayModelAdapter(ModelAdapter):
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

    def generate(self, package: PromptPackage, context: PromptContext) -> ModelResponse:
        if context.example.uid not in self.responses:
            raise KeyError(f"No replay response for uid={context.example.uid}")
        return ModelResponse(raw_text=self.responses[context.example.uid], metadata={"mode": "replay"})


class PlaceholderAgentAdapter(AgentAdapter):
    def generate(self, package: PromptPackage, context: PromptContext) -> ModelResponse:
        raise NotImplementedError(
            "Agent evaluation is reserved for a later phase. "
            "Implement a real AgentAdapter to use this provider."
        )
