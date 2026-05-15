import re
import json
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.graph.state import SupportState

load_dotenv()

logger = logging.getLogger(__name__)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def _parse_json_safe(text: str) -> dict:
    """Strip markdown fences then parse JSON — identical helper used in all agents."""
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(clean)


# ✅ ENHANCEMENT: Utilisation du PROMPT_B (détaillé) identifié comme meilleur
# lors de l'évaluation A/B dans prompt_tester.py. Le prompt court (PROMPT_A)
# omettait les exemples et les règles de format, causant des erreurs de parsing.
SYSTEM_PROMPT = """Tu es un agent de support client expérimenté. Ta tâche est d'analyser le message d'un client et d'en extraire deux informations :
1. L'intention principale : "suivi_commande" (veut savoir où en est sa commande), "demande_remboursement" (exige un remboursement), "réclamation" (se plaint d'un produit/service), ou "autre" (tout le reste).
2. Le numéro de commande si mentionné (par exemple #12345), sinon null.

Exemples :
- "Je n'ai pas reçu ma commande #12345, je veux être remboursé" → {"intent": "demande_remboursement", "order_id": "12345"}
- "Où est ma commande #98765 ?" → {"intent": "suivi_commande", "order_id": "98765"}
- "Le produit est cassé, que faire ?" → {"intent": "réclamation", "order_id": null}

IMPORTANT : réponds UNIQUEMENT au format JSON, sans texte supplémentaire ni balises markdown."""


def classifier_node(state: SupportState) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["query"]),
    ]

    response = llm.invoke(messages)

    # ✅ FIX: Utiliser _parse_json_safe pour gérer les blocs ```json
    # L'ancienne version utilisait json.loads() brut — si le LLM ajoutait des
    # backticks markdown, le parsing échouait silencieusement et retournait "autre".
    try:
        parsed = _parse_json_safe(response.content)
    except Exception as e:
        logger.warning(f"[classifier] parse failed: {e}. Raw: {response.content[:200]}")
        parsed = {"intent": "autre", "order_id": None}

    return {
        "intent": parsed.get("intent", "autre"),
        "order_id": parsed.get("order_id"),
    }