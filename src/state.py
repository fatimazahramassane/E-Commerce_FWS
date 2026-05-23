# state.py — État partagé du graphe LangGraph
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


def merge_dict(left: dict, right: dict) -> dict:
    """Fusionne deux dictionnaires pour les champs de type dict."""
    if not left:
        return right or {}
    if not right:
        return left
    return {**left, **right}


class AgentState(TypedDict, total=False):
    """État global traversé par les 5 nœuds du graphe."""

    messages: Annotated[list, add_messages]
    user_message: str
    intent: str
    order_id: Optional[str]
    confidence: float
    needs_human: bool
    classification_raw: str
    rag_context: str
    rag_chunks: list[str]
    human_payload: dict[str, Any]
    human_decision: Optional[str]
    human_approved: Optional[bool]
    resolver_notes: str
    final_response: str
    current_step: str
    error: Optional[str]
    workflow_log: list[str]
