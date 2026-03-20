from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    tier: str
    note: str
    source_url: str


# These are chosen as a practical starter pack for fragment-level benchmarking:
# a mix of value-oriented and strong frontier baselines chosen for this repo.
OPENROUTER_STARTER_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="deepseek/deepseek-chat-v3.1",
        family="DeepSeek",
        tier="value_open",
        note="Strong open-family baseline with good cost-performance.",
        source_url="https://openrouter.ai/deepseek/deepseek-chat-v3.1",
    ),
    ModelSpec(
        model_id="openai/gpt-5-mini",
        family="GPT",
        tier="value_closed",
        note="Compact GPT-5 baseline with strong instruction following at much lower cost than GPT-5.",
        source_url="https://openrouter.ai/openai/gpt-5-mini",
    ),
    ModelSpec(
        model_id="google/gemini-2.5-flash",
        family="Gemini",
        tier="workhorse_closed",
        note="Strong Google workhorse baseline with large context and reasoning support.",
        source_url="https://openrouter.ai/google/gemini-2.5-flash",
    ),
    ModelSpec(
        model_id="anthropic/claude-haiku-4.5",
        family="Claude",
        tier="fast_closed",
        note="Fast Anthropic baseline with strong general capability.",
        source_url="https://openrouter.ai/anthropic/claude-haiku-4.5",
    ),
    ModelSpec(
        model_id="openai/gpt-5",
        family="GPT",
        tier="frontier_closed",
        note="High-end frontier anchor for best-case capability comparisons.",
        source_url="https://openrouter.ai/openai/gpt-5",
    ),
)


MODEL_SETS: dict[str, tuple[ModelSpec, ...]] = {
    "starter": OPENROUTER_STARTER_MODELS,
}


def list_model_sets() -> list[str]:
    return sorted(MODEL_SETS)


def get_model_set(name: str) -> tuple[ModelSpec, ...]:
    try:
        return MODEL_SETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model set {name!r}. Available sets: {', '.join(list_model_sets())}"
        ) from exc
