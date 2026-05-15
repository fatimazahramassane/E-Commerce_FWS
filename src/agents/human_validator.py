from langgraph.types import interrupt
from src.graph.state import SupportState


def human_validation_node(state: SupportState) -> dict:
    """
    Ce nœud n'est atteint que si route_after_rag() retourne "human_validation",
    c'est-à-dire uniquement quand requires_human=True ET proposed_action="refund".
    La branche else de l'ancienne version était donc du code mort — supprimée.
    """
    # ✅ FIX: Suppression du if/else. La condition d'accès à ce nœud est déjà
    # garantie par le routage conditionnel dans workflow.py. Ajouter un second
    # if ici créait une illusion de sécurité et une branche else jamais exécutée.
    decision = interrupt({
        "message": f"Action de remboursement demandée pour la commande {state.get('order_id')}.",
        "details": state.get("proposed_response"),
        "action": "refund",
    })

    if decision.get("approved"):
        return {"approved": True}
    else:
        return {
            "approved": False,
            "final_response": (
                "Après vérification, nous ne pouvons pas procéder au remboursement "
                "pour le moment. Veuillez contacter le support pour plus d'informations."
            ),
        }