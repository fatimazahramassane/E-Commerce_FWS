# src/ui/app.py — Interface Streamlit complète (workflow, RAG, HITL, historique)
"""
Lancement : streamlit run src/ui/app.py
Depuis la racine du projet après pip install -r requirements.txt
"""
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

# Racine projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.graph import get_compiled_graph, get_thread_config  # noqa: E402
from src.rag.ingest import ingest_policy  # noqa: E402

# ——— Configuration page ———
st.set_page_config(
    page_title="Support Client E-commerce — Multi-Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

HITL_TIMEOUT_SECONDS = 300


def init_session_state():
    """Initialise thread_id persistant et historique (critère checkpointer RETAL)."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_hitl" not in st.session_state:
        st.session_state.pending_hitl = None
    if "last_snapshot" not in st.session_state:
        st.session_state.last_snapshot = None
    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False


def ensure_rag_index():
    """Ingère la politique au premier chargement."""
    if not st.session_state.rag_ready:
        with st.spinner("Indexation RAG (ChromaDB + embeddings)…"):
            try:
                ingest_policy(force=False)
                st.session_state.rag_ready = True
            except Exception as exc:
                st.error(f"Erreur ingestion RAG : {exc}")


def render_sidebar():
    """Barre latérale : config, thread, reset."""
    st.sidebar.header("Configuration")
    api_ok = bool(__import__("os").environ.get("GROQ_API_KEY"))
    st.sidebar.markdown(
        f"**GROQ_API_KEY** : {' Définie' if api_ok else ' Manquante'}"
    )
    st.sidebar.text_input(
        "Thread ID (checkpointer)",
        value=st.session_state.thread_id,
        disabled=True,
        help="Identifiant persistant MemorySaver pour reprise HITL",
    )
    if st.sidebar.button("🔄 Nouvelle conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.pending_hitl = None
        st.session_state.last_snapshot = None
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Workflow LangGraph**
        1. Classifier (Pydantic)
        2. RAG Verifier
        3. HITL (si remboursement + order_id)
        4. Resolver
        5. Responder
        """
    )


def extract_interrupt_payload(snapshot) -> dict | None:
    """Extrait le payload HITL depuis l'état LangGraph."""
    if not snapshot:
        return None
    interrupts = getattr(snapshot, "interrupts", None) or []
    if interrupts:
        val = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        return val if isinstance(val, dict) else {"raw": val}
    # Fallback via tasks (anciennes versions)
    tasks = getattr(snapshot, "tasks", None) or []
    for task in tasks:
        if getattr(task, "interrupts", None):
            intr = task.interrupts[0]
            return intr.value if hasattr(intr, "value") else intr
    return None


def run_graph_input(user_message: str, is_resume: bool = False, resume_value=None):
    """Invoque le graphe (nouveau message ou reprise HITL)."""
    graph = get_compiled_graph()
    config = get_thread_config(st.session_state.thread_id)

    if is_resume and resume_value is not None:
        return graph.invoke(Command(resume=resume_value), config=config)

    return graph.invoke(
        {"user_message": user_message, "workflow_log": []},
        config=config,
    )


def process_user_message(user_message: str):
    """Traite un message utilisateur et gère les interruptions HITL."""
    try:
        result = run_graph_input(user_message)
        graph = get_compiled_graph()
        config = get_thread_config(st.session_state.thread_id)
        snapshot = graph.get_state(config)
        st.session_state.last_snapshot = snapshot

        payload = extract_interrupt_payload(snapshot)
        if payload:
            st.session_state.pending_hitl = payload
            return None, payload

        final = result.get("final_response") if isinstance(result, dict) else None
        if not final and snapshot and snapshot.values:
            final = snapshot.values.get("final_response")
        return final, None
    except Exception as exc:
        st.error(f"Erreur exécution graphe : {exc}")
        return None, None


def resume_hitl(decision: str):
    """Reprise via Command(resume=...) — critère HITL RETAL."""
    approved = decision == "approved"
    resume_payload = {"decision": decision, "approved": approved}
    try:
        graph = get_compiled_graph()
        config = get_thread_config(st.session_state.thread_id)
        result = graph.invoke(Command(resume=resume_payload), config=config)
        snapshot = graph.get_state(config)
        st.session_state.last_snapshot = snapshot
        st.session_state.pending_hitl = None

        final = result.get("final_response") if isinstance(result, dict) else None
        if not final and snapshot.values:
            final = snapshot.values.get("final_response")
        return final
    except Exception as exc:
        st.error(f"Erreur reprise HITL : {exc}")
        return None


def render_workflow_expanders(snapshot):
    """Expanders temps réel : étape, RAG, HITL, réponse."""
    values = snapshot.values if snapshot and snapshot.values else {}

    with st.expander("📍 Étape actuelle", expanded=True):
        st.write(values.get("current_step", "—"))
        logs = values.get("workflow_log") or []
        if logs:
            st.code("\n".join(logs[-8:]), language=None)

    with st.expander("📚 Contexte RAG"):
        chunks = values.get("rag_chunks") or []
        st.caption(f"{len(chunks)} chunk(s) récupéré(s) (top_k=3)")
        st.text_area(
            "Contexte injecté",
            value=values.get("rag_context", "—"),
            height=150,
            disabled=True,
        )

    with st.expander("👤 Décision HITL"):
        if values.get("human_decision"):
            approved = values.get("human_approved")
            st.success("Approuvé") if approved else st.warning("Refusé")
            st.json(
                {
                    "decision": values.get("human_decision"),
                    "approved": approved,
                    "payload": values.get("human_payload"),
                }
            )
        else:
            st.info("Aucune validation humaine sur ce tour (bypass ou en attente).")

    with st.expander("✅ Classification (Pydantic)"):
        st.json(
            {
                "intent": values.get("intent"),
                "order_id": values.get("order_id"),
                "confidence": values.get("confidence"),
                "needs_human": values.get("needs_human"),
                "error": values.get("error"),
            }
        )


def render_hitl_panel(payload: dict):
    """UI HITL : boutons Approuver / Refuser + payload + timeout."""
    st.warning("⏸️ Validation financière requise (Human-in-the-Loop)")
    st.markdown(
        f"**Commande** : `{payload.get('order_id', 'N/A')}`  \n"
        f"**Message client** : {payload.get('user_message', '')[:200]}"
    )
    st.info(payload.get("message", "Validation remboursement"))
    st.caption(
        f"Timeout suggéré : {payload.get('timeout_seconds', HITL_TIMEOUT_SECONDS)}s — "
        f"Horodatage : {payload.get('timestamp_utc', 'N/A')}"
    )

    with st.expander("Payload complet à valider"):
        st.json(payload)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approuver", type="primary", use_container_width=True):
            final = resume_hitl("approved")
            if final:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": final, "hitl": "approved"}
                )
            st.rerun()
    with col2:
        if st.button("❌ Refuser", use_container_width=True):
            final = resume_hitl("refused")
            if final:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": final, "hitl": "refused"}
                )
            st.rerun()


def main():
    init_session_state()
    ensure_rag_index()
    render_sidebar()

    st.title(" Support Client E-commerce -- F W S -- ")
    st.markdown(
        "Suivi commande · Remboursement · Réclamation — "
        "LangGraph + RAG + HITL "
    )

    # Historique
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("hitl"):
                st.caption(f"Décision HITL : {entry['hitl']}")

    # Panneau HITL en attente
    if st.session_state.pending_hitl:
        render_hitl_panel(st.session_state.pending_hitl)

    # Saisie utilisateur
    user_input = st.chat_input("Décrivez votre demande (ex: remboursement ORD-12345)…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Traitement par le graphe multi-agent…"):
            final, hitl_payload = process_user_message(user_input)

        if hitl_payload:
            st.session_state.pending_hitl = hitl_payload
            render_hitl_panel(hitl_payload)
        elif final:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": final}
            )
            with st.chat_message("assistant"):
                st.markdown(final)

    # Expanders workflow (dernier état)
    snapshot = st.session_state.last_snapshot
    if snapshot:
        st.markdown("---")
        st.subheader("Détail du workflow")
        render_workflow_expanders(snapshot)

        values = snapshot.values or {}
        if values.get("final_response") and not st.session_state.pending_hitl:
            st.success("Réponse finale")
            st.markdown(values["final_response"])


if __name__ == "__main__":
    main()
