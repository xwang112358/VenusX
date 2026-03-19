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
# strong enough to be meaningful, cheaper than flagship models, and drawn from
# model families that commonly appear in current LLM benchmark papers.
OPENROUTER_STARTER_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="google/gemini-2.5-flash-lite",
        family="Gemini",
        tier="budget_closed",
        note="Very cheap closed-model anchor with long context and good structured output behavior.",
        source_url="https://openrouter.ai/google/gemini-2.5-flash-lite",
    ),
    ModelSpec(
        model_id="openai/gpt-4.1-mini",
        family="GPT",
        tier="budget_closed",
        note="Good anchor closed-source baseline with strong instruction following.",
        source_url="https://openrouter.ai/openai/gpt-4.1-mini",
    ),
    ModelSpec(
        model_id="deepseek/deepseek-chat-v3.1",
        family="DeepSeek",
        tier="budget_open",
        note="Strong open-family reasoning baseline that stays relatively cheap on OpenRouter.",
        source_url="https://openrouter.ai/deepseek/deepseek-chat-v3.1",
    ),
    ModelSpec(
        model_id="meta-llama/llama-3.3-70b-instruct",
        family="Llama",
        tier="budget_open",
        note="Widely used open-family baseline with low cost on OpenRouter.",
        source_url="https://openrouter.ai/meta-llama/llama-3.3-70b-instruct",
    ),
    ModelSpec(
        model_id="qwen/qwen-2.5-72b-instruct",
        family="Qwen",
        tier="budget_open",
        note="Strong open-family baseline that is common in benchmark comparisons and usually behaves well with JSON.",
        source_url="https://openrouter.ai/qwen/qwen-2.5-72b-instruct",
    ),
)


OPENROUTER_EXTENDED_MODELS: tuple[ModelSpec, ...] = OPENROUTER_STARTER_MODELS + (
    ModelSpec(
        model_id="anthropic/claude-3.5-haiku",
        family="Claude",
        tier="extra_closed",
        note="Useful Anthropic anchor baseline, but notably pricier on output tokens than the starter pack.",
        source_url="https://openrouter.ai/anthropic/claude-3.5-haiku",
    ),
    ModelSpec(
        model_id="mistralai/mistral-small-3.2-24b-instruct",
        family="Mistral",
        tier="extra_open",
        note="Very cheap smaller open-model baseline with strong structured-output behavior.",
        source_url="https://openrouter.ai/mistralai/mistral-small-3.2-24b-instruct",
    ),
)


MODEL_SETS: dict[str, tuple[ModelSpec, ...]] = {
    "starter": OPENROUTER_STARTER_MODELS,
    "extended": OPENROUTER_EXTENDED_MODELS,
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
