"""Protein agent — OpenAI tool-use agent for InterPro site annotation."""
from protein_agent.agent import ProteinAgent
from protein_agent.records import AgentResult, SiteAnnotation

__all__ = ["ProteinAgent", "AgentResult", "SiteAnnotation"]


def __getattr__(name: str):
    if name == "ProteinAgent":
        from protein_agent.agent import ProteinAgent

        return ProteinAgent
    if name == "agent":
        return importlib.import_module("protein_agent.agent")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
