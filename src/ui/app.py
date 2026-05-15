import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
from src.graph.workflow import graph, new_thread_id
from langgraph.types import Command

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Support Client Intelligent", layout="wide", page_icon="🤖")
st.title("🤖 Support Client Intelligent")

# ── Session state init ─────────────────────────────────────────────────────────
def _init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = new_thread_id()
    if "events" not in st.session_state:
        st.session_state.events = []
    if "interrupted" not in st.session_state:
        st.session_state.interrupted = False
    if "interrupt_payload" not in st.session_state:
        st.session_state.interrupt_payload = None
    # ✅ ENHANCEMENT: Stocker la réponse finale séparément pour l'afficher en évidence
    if "final_response" not in st.session_state:
        st.session_state.final_response = None

_init_session()

# ── Helpers ────────────────────────────────────────────────────────────────────
def _config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}

def _run_sync(input_or_command):
    """Run graph synchronously and collect events + final response."""
    events = []
    interrupted = False
    interrupt_payload = None
    final_response = None

    for event in graph.stream(input_or_command, _config(), stream_mode="updates"):
        if "__interrupt__" in event:
            interrupted = True
            interrupt_payload = event["__interrupt__"]
            break
        events.append(event)

        # ✅ ENHANCEMENT: Extraire la réponse finale dès qu'elle apparaît
        for node_data in event.values():
            if isinstance(node_data, dict) and node_data.get("final_response"):
                final_response = node_data["final_response"]

    return events, interrupted, interrupt_payload, final_response

def _display_events(events):
    """Affiche les étapes du workflow de façon claire."""
    NODE_LABELS = {
        "classifier":       ("🏷️", "Classification de l'intention"),
        "rag_verifier":     ("📚", "Vérification RAG"),
        "human_validation": ("👤", "Validation humaine"),
        "resolver":         ("✍️", "Génération de la réponse"),
        "responder":        ("📤", "Envoi de la réponse"),
    }
    for evt in events:
        for node, data in evt.items():
            if node.startswith("_"):
                continue
            icon, label = NODE_LABELS.get(node, ("📌", node))
            with st.expander(f"{icon} {label}", expanded=False):
                st.json(data)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📝 Nouvelle demande")
    user_query = st.text_area(
        "Message du client :",
        height=120,
        placeholder="Ex : Je n'ai pas reçu ma commande #12345, je veux être remboursé",
    )

    col1, col2 = st.columns(2)
    with col1:
        submit = st.button("Envoyer", type="primary", use_container_width=True)
    with col2:
        if st.button("Réinitialiser", use_container_width=True):
            st.session_state.thread_id = new_thread_id()
            st.session_state.events = []
            st.session_state.interrupted = False
            st.session_state.interrupt_payload = None
            st.session_state.final_response = None
            st.rerun()

    st.markdown("---")
    st.caption(f"Thread : `{st.session_state.thread_id}`")

# ── Main area ──────────────────────────────────────────────────────────────────

if submit and user_query.strip():
    st.session_state.events = []
    st.session_state.interrupted = False
    st.session_state.interrupt_payload = None
    st.session_state.final_response = None

    initial_state = {
        "query": user_query.strip(),
        "intent": None,
        "order_id": None,
        "retrieved_context": None,
        "proposed_response": None,
        "proposed_action": None,
        "requires_human": False,
        "final_response": None,
        "approved": None,
        "error": None,
    }

    with st.spinner("Traitement en cours…"):
        new_events, interrupted, payload, final = _run_sync(initial_state)

    st.session_state.events.extend(new_events)
    st.session_state.interrupted = interrupted
    st.session_state.interrupt_payload = payload
    if final:
        st.session_state.final_response = final

    st.rerun()

# ── Workflow steps ─────────────────────────────────────────────────────────────
if st.session_state.events or st.session_state.interrupted:
    st.header("🔄 Déroulement du traitement")
    _display_events(st.session_state.events)

# ── Final response banner ──────────────────────────────────────────────────────
# ✅ ENHANCEMENT: La réponse finale est maintenant affichée de façon proéminente
# plutôt que d'être enfouie dans un expander JSON du nœud "responder".
if st.session_state.final_response and not st.session_state.interrupted:
    st.markdown("---")
    st.header("💬 Réponse envoyée au client")
    st.success(st.session_state.final_response)

# ── Human-in-the-loop panel ────────────────────────────────────────────────────
if st.session_state.interrupted and st.session_state.interrupt_payload:
    st.markdown("---")
    st.header("🔐 Validation humaine requise")

    payload = st.session_state.interrupt_payload
    details = {}
    try:
        for item in payload:
            val = item.value if hasattr(item, "value") else item
            if isinstance(val, dict):
                details = val
                break
    except Exception:
        pass

    if details:
        st.warning(f"**{details.get('message', 'Action critique détectée')}**")
        st.write(f"**Détails :** {details.get('details', 'N/A')}")
        st.write(f"**Action :** `{details.get('action', 'refund')}`")
    else:
        st.warning("Une validation humaine est requise pour continuer.")

    col1, col2, _ = st.columns([1, 1, 3])

    with col1:
        if st.button("✅ Approuver", type="primary"):
            with st.spinner("Reprise…"):
                new_events, interrupted, payload, final = _run_sync(
                    Command(resume={"approved": True})
                )
            st.session_state.events.extend(new_events)
            st.session_state.interrupted = interrupted
            st.session_state.interrupt_payload = payload
            if final:
                st.session_state.final_response = final
            if not interrupted:
                st.success("✅ Remboursement approuvé — traitement terminé.")
            st.rerun()

    with col2:
        if st.button("❌ Refuser"):
            with st.spinner("Reprise…"):
                new_events, interrupted, payload, final = _run_sync(
                    Command(resume={"approved": False})
                )
            st.session_state.events.extend(new_events)
            st.session_state.interrupted = interrupted
            st.session_state.interrupt_payload = payload
            if final:
                st.session_state.final_response = final
            if not interrupted:
                st.info("Demande refusée — réponse envoyée au client.")
            st.rerun()

elif not st.session_state.events and not st.session_state.interrupted:
    st.info("Entrez un message dans la barre latérale et cliquez sur **Envoyer** pour démarrer.")