import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.graph.state import SupportState
from src.agents.classifier import classifier_node
from src.agents.verifier_rag import verifier_rag_node
from src.agents.resolver import resolver_node
from src.agents.human_validator import human_validation_node
from src.agents.responder import responder_node


def route_after_rag(state: SupportState) -> str:
    """Route vers validation humaine si remboursement critique, sinon au resolver."""
    if state.get("requires_human") and state.get("proposed_action") == "refund":
        return "human_validation"
    return "resolver"


# ✅ FIX: Suppression de route_after_human — définie mais jamais utilisée.
# L'edge human_validation→resolver était un edge fixe, pas conditionnel.
# Garder du code mort induit les relecteurs en erreur sur le flux réel.

builder = StateGraph(SupportState)

builder.add_node("classifier", classifier_node)
builder.add_node("rag_verifier", verifier_rag_node)
builder.add_node("resolver", resolver_node)
builder.add_node("human_validation", human_validation_node)
builder.add_node("responder", responder_node)

builder.add_edge(START, "classifier")
builder.add_edge("classifier", "rag_verifier")

builder.add_conditional_edges(
    "rag_verifier",
    route_after_rag,
    {
        "human_validation": "human_validation",
        "resolver": "resolver",
    },
)

builder.add_edge("human_validation", "resolver")
builder.add_edge("resolver", "responder")
builder.add_edge("responder", END)

memory = MemorySaver()

# ✅ FIX: Suppression de interrupt_before=["human_validation"].
# human_validation_node appelle interrupt() en interne — utiliser les deux
# causait une double pause : la première consommait le Command(resume=...)
# de l'UI et la seconde ne recevait jamais l'approbation.
graph = builder.compile(checkpointer=memory)


def new_thread_id() -> str:
    """Génère un identifiant de thread unique. Utilisé par app.py."""
    return f"thread_{uuid.uuid4().hex[:8]}"