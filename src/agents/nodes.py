# src/agents/nodes.py — 5 nœuds LangGraph : classifier, rag_verifier, human_interrupt, resolver, responder
"""
Nœuds du graphe multi-agent support client e-commerce.
- classifier : validation JSON stricte via Pydantic
- rag_verifier : retrieval ChromaDB top_k=3
- human_interrupt : interrupt() conditionnel (validation financière)
- resolver : synthèse avec contexte RAG
- responder : réponse finale client
"""
import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.types import interrupt

from src.llm import extract_json_from_text, get_groq_llm, invoke_llm_safe
from src.prompts import PROMPT_A_CLASSIFIER, PROMPT_B_CLASSIFIER, RESOLVER_PROMPT, RESPONDER_PROMPT
from src.rag.ingest import retrieve_context
from src.schemas import ClassificationOutput
from src.state import AgentState

logger = logging.getLogger(__name__)

# Prompt actif pour le graphe (B par défaut — plus structuré)
ACTIVE_CLASSIFIER_PROMPT = PROMPT_B_CLASSIFIER

HITL_TIMEOUT_SECONDS = 300  # 5 minutes — affiché dans l'UI


def _append_log(state: AgentState, message: str) -> list[str]:
    logs = list(state.get("workflow_log") or [])
    logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}")
    return logs


def classifier_node(state: AgentState) -> dict[str, Any]:
    """
    Nœud 1 — Classification de l'intention avec validation Pydantic stricte.
    Évite les crashes en cas de JSON invalide du LLM.
    """
    user_message = state.get("user_message") or ""
    if not user_message and state.get("messages"):
        last = state["messages"][-1]
        user_message = getattr(last, "content", str(last))

    updates: dict[str, Any] = {
        "current_step": "classifier",
        "workflow_log": _append_log(state, "Démarrage classification"),
    }

    try:
        llm = get_groq_llm(temperature=0.0)
        prompt = ACTIVE_CLASSIFIER_PROMPT.format(user_message=user_message)
        raw_text, err = invoke_llm_safe(llm, prompt)
        updates["classification_raw"] = raw_text

        if err:
            updates["error"] = err
            updates["intent"] = "autre"
            updates["order_id"] = None
            updates["confidence"] = 0.0
            updates["needs_human"] = False
            updates["workflow_log"] = _append_log(
                {**state, **updates}, f"Échec LLM classifier: {err}"
            )
            return updates

        parsed = extract_json_from_text(raw_text)
        if not parsed:
            updates["error"] = "JSON invalide du classifier"
            updates["intent"] = "autre"
            updates["order_id"] = None
            updates["confidence"] = 0.0
            updates["needs_human"] = False
            return updates

        # Validation Pydantic stricte
        validated = ClassificationOutput.model_validate(parsed)
        # Règle métier alignée sur le routage HITL (remboursement + order_id)
        enforced_human = (
            validated.intent == "demande_remboursement" and validated.order_id is not None
        )
        updates["intent"] = validated.intent
        updates["order_id"] = validated.order_id
        updates["confidence"] = validated.confidence
        updates["needs_human"] = enforced_human
        updates["error"] = None
        updates["workflow_log"] = _append_log(
            {**state, **updates},
            f"Intent={validated.intent}, order_id={validated.order_id}, needs_human={validated.needs_human}",
        )
    except Exception as exc:
        logger.exception("Erreur classifier_node")
        updates["error"] = str(exc)
        updates["intent"] = "autre"
        updates["order_id"] = None
        updates["needs_human"] = False

    return updates


def rag_verifier_node(state: AgentState) -> dict[str, Any]:
    """
    Nœud 2 — Vérification RAG agentique : retrieval top_k=3, injection pour resolver.
    Fallback standard si aucun chunk.
    """
    user_message = state.get("user_message") or ""
    intent = state.get("intent") or "autre"
    query = f"{intent} {user_message}"
    if state.get("order_id"):
        query += f" {state['order_id']}"

    context, chunks = retrieve_context(query, top_k=3)

    return {
        "current_step": "rag_verifier",
        "rag_context": context,
        "rag_chunks": chunks,
        "workflow_log": _append_log(
            state,
            f"RAG: {len(chunks)} chunk(s) récupéré(s)",
        ),
    }


def human_interrupt_node(state: AgentState) -> dict[str, Any]:
    """
    Nœud 3 — Human-in-the-Loop via interrupt().
    Déclenché uniquement sur routage conditionnel (remboursement + order_id).
    Reprise : Command(resume=...) → valeur retournée par interrupt().
    """
    payload = {
        "type": "validation_financiere",
        "intent": state.get("intent"),
        "order_id": state.get("order_id"),
        "user_message": state.get("user_message"),
        "rag_excerpt": (state.get("rag_context") or "")[:400],
        "message": (
            f"Validation requise pour remboursement — Commande {state.get('order_id')}. "
            "Article 5 : commandes > 500€ nécessitent validation financière."
        ),
        "timeout_seconds": HITL_TIMEOUT_SECONDS,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # interrupt() suspend le graphe — reprise via Command(resume=...)
    human_decision = interrupt(payload)

    approved = False
    decision_str = "refused"
    if isinstance(human_decision, dict):
        decision_str = human_decision.get("decision", "refused")
        approved = decision_str in ("approved", "approuver", "approve", True)
    elif isinstance(human_decision, bool):
        approved = human_decision
        decision_str = "approved" if approved else "refused"
    elif isinstance(human_decision, str):
        decision_str = human_decision.lower()
        approved = decision_str in ("approved", "approuver", "approve", "oui", "yes", "true")

    return {
        "current_step": "human_interrupt",
        "human_payload": payload,
        "human_decision": decision_str,
        "human_approved": approved,
        "workflow_log": _append_log(
            state,
            f"HITL terminé: decision={decision_str}, approved={approved}",
        ),
    }


def resolver_node(state: AgentState) -> dict[str, Any]:
    """
    Nœud 4 — Résolution avec injection du contexte RAG et décision HITL si présente.
    """
    intent = state.get("intent") or "autre"
    order_id = state.get("order_id") or "N/A"
    human_decision = state.get("human_decision") or "non applicable"
    rag_context = state.get("rag_context") or ""

    notes = ""
    try:
        llm = get_groq_llm(temperature=0.2)
        prompt = RESOLVER_PROMPT.format(
            intent=intent,
            order_id=order_id,
            human_decision=human_decision,
            rag_context=rag_context,
        )
        notes, err = invoke_llm_safe(llm, prompt)
        if err:
            notes = f"Résolution locale: intent={intent}, order={order_id}. {err}"
    except Exception as exc:
        notes = f"Résolution de secours: {exc}"

    if state.get("human_approved") is False:
        notes += " [HITL: remboursement refusé par validateur humain]"

    return {
        "current_step": "resolver",
        "resolver_notes": notes,
        "workflow_log": _append_log(state, "Résolution terminée"),
    }


def responder_node(state: AgentState) -> dict[str, Any]:
    """
    Nœud 5 — Génération de la réponse finale au client.
    """
    intent = state.get("intent") or "autre"
    order_id = state.get("order_id") or "non fourni"
    human_approved = state.get("human_approved")
    resolver_notes = state.get("resolver_notes") or ""
    rag_context = state.get("rag_context") or ""

    final = ""
    try:
        llm = get_groq_llm(temperature=0.3)
        prompt = RESPONDER_PROMPT.format(
            intent=intent,
            order_id=order_id,
            human_approved=human_approved,
            resolver_notes=resolver_notes,
            rag_context=rag_context[:800],
        )
        final, err = invoke_llm_safe(llm, prompt)
        if err or not final:
            final = _fallback_response(state, err)
    except Exception as exc:
        final = _fallback_response(state, str(exc))

    return {
        "current_step": "responder",
        "final_response": final,
        "workflow_log": _append_log(state, "Réponse finale générée"),
    }


def _fallback_response(state: AgentState, err: str | None) -> str:
    """Réponse de secours si le LLM échoue."""
    intent = state.get("intent", "autre")
    oid = state.get("order_id") or "votre commande"
    if intent == "suivi_commande":
        return (
            f"Merci pour votre message. Nous traitons le suivi de {oid}. "
            "Un conseiller vous répondra sous 24h. (Mode secours)"
        )
    if intent == "demande_remboursement":
        if state.get("human_approved") is False:
            return (
                f"Votre demande de remboursement pour {oid} n'a pas été approuvée "
                "par notre service financier. Contactez support@boutique-sdia.fr."
            )
        return (
            f"Votre demande de remboursement pour {oid} est en cours d'examen. "
            "Délai standard : 7 jours ouvrés après réception du retour."
        )
    return (
        "Merci de nous avoir contactés. Notre équipe traite votre demande. "
        f"{('Erreur technique: ' + err) if err else ''}"
    )


def classify_with_prompt(user_message: str, prompt_template: str) -> tuple[ClassificationOutput | None, str, bool]:
    """
    Utilitaire pour evaluation.py — teste Prompt A ou B.
    Retourne (ClassificationOutput|None, raw_text, json_valid).
    """
    raw = ""
    try:
        llm = get_groq_llm(temperature=0.0)
        prompt = prompt_template.format(user_message=user_message)
        raw, err = invoke_llm_safe(llm, prompt)
        if err:
            return None, raw, False
        parsed = extract_json_from_text(raw)
        if not parsed:
            return None, raw, False
        validated = ClassificationOutput.model_validate(parsed)
        return validated, raw, True
    except Exception:
        return None, raw, False
