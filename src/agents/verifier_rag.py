import re
import json
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.graph.state import SupportState
from src.tools.rag_tool import retrieve_context

logger = logging.getLogger(__name__)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def _parse_json_safe(text: str) -> dict:
    """Strip markdown fences then parse JSON."""
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(clean)

def verifier_rag_node(state: SupportState) -> dict:
    context = retrieve_context(state["query"])

    system_prompt = f"""Tu es un expert en support client. Utilise UNIQUEMENT le contexte ci-dessous pour répondre.
Si l'information n'est pas présente dans le contexte, indique-le poliment sans inventer.

<contexte>
{context}
</contexte>

Analyse la demande du client (intention={state['intent']}, commande={state.get('order_id')}) et fournis :
- "rag_answer" : réponse factuelle tirée du contexte.
- "needs_human" : true uniquement si une action de remboursement réelle doit être déclenchée, sinon false.
- "action" : "refund" | "info" | "escalate" selon le cas.

IMPORTANT : réponds UNIQUEMENT au format JSON sans balises markdown.
Format : {{"rag_answer": "...", "needs_human": false, "action": "info"}}"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state["query"])]
    response = llm.invoke(messages)

    try:
        parsed = _parse_json_safe(response.content)
    except Exception as e:
        logger.warning(f"[verifier_rag] parse failed: {e}. Raw: {response.content[:200]}")
        parsed = {
            "rag_answer": "Désolé, je ne peux pas traiter cette demande pour le moment.",
            "needs_human": False,
            "action": "info"
        }

    return {
        "retrieved_context": context,                            # str, aligné avec state.py
        "proposed_response": parsed.get("rag_answer", ""),
        "proposed_action": parsed.get("action", "info"),
        "requires_human": bool(parsed.get("needs_human", False))  # force bool
    }