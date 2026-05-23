# src/graph.py — StateGraph LangGraph, routage conditionnel HITL, MemorySaver
"""
Architecture :
  classifier → rag_verifier → [human_interrupt | resolver] → resolver → responder → END

Routage conditionnel (critère HITL RETAL) :
  si intent == "demande_remboursement" ET order_id présent → human_interrupt
  sinon → bypass direct vers resolver
"""
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.nodes import (
    classifier_node,
    human_interrupt_node,
    rag_verifier_node,
    resolver_node,
    responder_node,
)
from src.state import AgentState

# Instance globale du checkpointer (persistance par thread_id)
_checkpointer = MemorySaver()
_compiled_graph = None


def route_after_rag(state: AgentState) -> Literal["human_interrupt", "resolver"]:
    """
    Routage conditionnel post-RAG :
    Validation financière humaine uniquement pour remboursement avec order_id.
    """
    intent = state.get("intent") or ""
    order_id = state.get("order_id")
    if intent == "demande_remboursement" and order_id:
        return "human_interrupt"
    return "resolver"


def build_graph():
    """Construit et compile le StateGraph avec 5 nœuds."""
    builder = StateGraph(AgentState)

    # ——— Enregistrement des 5 nœuds ———
    builder.add_node("classifier", classifier_node)
    builder.add_node("rag_verifier", rag_verifier_node)
    builder.add_node("human_interrupt", human_interrupt_node)
    builder.add_node("resolver", resolver_node)
    builder.add_node("responder", responder_node)

    # ——— Chaîne principale ———
    builder.set_entry_point("classifier")
    builder.add_edge("classifier", "rag_verifier")

    # Routage conditionnel HITL vs bypass
    builder.add_conditional_edges(
        "rag_verifier",
        route_after_rag,
        {
            "human_interrupt": "human_interrupt",
            "resolver": "resolver",
        },
    )

    builder.add_edge("human_interrupt", "resolver")
    builder.add_edge("resolver", "responder")
    builder.add_edge("responder", END)

    return builder.compile(checkpointer=_checkpointer)


def get_compiled_graph():
    """Singleton du graphe compilé."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def get_thread_config(thread_id: str) -> dict:
    """Configuration LangGraph avec thread_id persistant (Streamlit session_state)."""
    return {"configurable": {"thread_id": thread_id}}
