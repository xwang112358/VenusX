"""Protein agent package exports.

Keep ``ProteinAgent`` lazily imported so modules like ``protein_agent.records``
can be used in tests without requiring the optional ``anthropic`` dependency.
"""
import importlib

from protein_agent.records import AgentResult, SiteAnnotation

__all__ = ["ProteinAgent", "AgentResult", "SiteAnnotation"]


def __getattr__(name: str):
    if name == "ProteinAgent":
        from protein_agent.agent import ProteinAgent

        return ProteinAgent
    if name == "agent":
        return importlib.import_module("protein_agent.agent")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
