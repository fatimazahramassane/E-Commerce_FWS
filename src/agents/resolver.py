import re
import json
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.graph.state import SupportState

logger = logging.getLogger(__name__)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def _parse_json_safe(text: str) -> dict:
    """Strip markdown fences then parse JSON."""
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(clean)

def resolver_node(state: SupportState) -> dict:
    approved = state.get("approved", True)
    action = state.get("proposed_action", "info")

    system_prompt = f"""Tu es un assistant de support client aimable.
Rédige une réponse finale à envoyer au client en te basant sur les éléments ci-dessous.

Intention : {state['intent']}
Commande concernée : {state.get('order_id')}
Réponse factuelle fournie par l'expert : {state.get('proposed_response')}
Action proposée : {action}
Remboursement approuvé par un humain : {approved}

Règles :
- Si action='refund' ET approuvé=True : confirme que le remboursement sera traité sous 5-7 jours ouvrés.
- Si action='refund' ET approuvé=False : explique poliment que la demande ne peut pas aboutir et invite à contacter le support.
- Si action='info' : fournis les informations de façon claire et concise.
- Si action='escalate' : explique qu'un conseiller humain va prendre le relais sous 24h.

IMPORTANT : réponds UNIQUEMENT au format JSON sans balises markdown.
Format : {{"final_response": "..."}}"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state["query"])]
    response = llm.invoke(messages)

    try:
        parsed = _parse_json_safe(response.content)
        final = parsed.get("final_response", "")
        if not final:
            raise ValueError("empty final_response")
    except Exception as e:
        logger.warning(f"[resolver] parse failed: {e}")
        final = state.get("proposed_response", "Nous allons traiter votre demande dans les plus brefs délais.")

    return {"final_response": final}