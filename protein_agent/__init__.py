"""Protein agent — Claude tool-use agent for InterPro site annotation."""
from protein_agent.agent import ProteinAgent
from protein_agent.records import AgentResult, SiteAnnotation

__all__ = ["ProteinAgent", "AgentResult", "SiteAnnotation"]
