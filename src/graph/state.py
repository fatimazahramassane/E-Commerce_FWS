from typing import TypedDict, Optional


class SupportState(TypedDict):
    query: str                          # message original du client
    intent: Optional[str]               # classification agent output
    order_id: Optional[str]             # extrait de la commande si présente
    retrieved_context: Optional[str]    # texte brut retourné par le RAG
    proposed_response: Optional[str]    # proposition de réponse
    proposed_action: Optional[str]      # "refund", "info", "escalate"
    requires_human: Optional[bool]      # pour interruption
    final_response: Optional[str]
    approved: Optional[bool]            # décision humaine
    # ✅ ENHANCEMENT: Champ error pour tracer les échecs de nœuds sans planter
    # le workflow. Permet à l'UI d'afficher un message d'erreur clair.
    error: Optional[str]