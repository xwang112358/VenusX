from __future__ import annotations

from abc import ABC, abstractmethod

from evaluation_llm.catalog import LabelCatalog
from evaluation_llm.types import (
    CandidateRecord,
    EvaluationRecord,
    FragmentExample,
    ModelResponse,
    ParsedPrediction,
    PromptContext,
    PromptPackage,
)


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, package: PromptPackage, context: PromptContext) -> ModelResponse:
        raise NotImplementedError


class AgentAdapter(ModelAdapter):
    """Placeholder interface for future agent-style evaluation."""


class CandidateProvider(ABC):
    @abstractmethod
    def get_candidates(self, example: FragmentExample, top_k: int | None = None) -> list[CandidateRecord]:
        raise NotImplementedError

    def get_few_shots(
        self,
        example: FragmentExample,
        candidate_ids: list[str],
        limit: int,
    ) -> list[FragmentExample]:
        return []


class PromptBuilder(ABC):
    @abstractmethod
    def build(
        self,
        context: PromptContext,
        catalog: LabelCatalog,
    ) -> PromptPackage:
        raise NotImplementedError


class ResponseParser(ABC):
    @abstractmethod
    def parse(self, raw_text: str, catalog: LabelCatalog) -> ParsedPrediction:
        raise NotImplementedError


class MetricSuite(ABC):
    @abstractmethod
    def update(self, record: EvaluationRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> dict:
        raise NotImplementedError
