from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. DEFINE THE STATE (Shared Memory)
# ==========================================
class SystemState(TypedDict):
    messages: Annotated[list, add_messages]
    proposed_action: str
    human_approved: bool

# ==========================================
# 2. DEFINE THE NODES (Dummy Agents)
# ==========================================
def planner_agent(state: SystemState):
    """Simulates an agent diagnosing an issue and proposing a high-risk action."""
    print("🤖 Agent Diagnostiqueur: J'ai détecté une anomalie critique (Fraude/Panne).")
    print("🤖 Agent Diagnostiqueur: Je demande la permission de bloquer le système.")
    
    return {
        "messages": ["Demande de blocage initiée."],
        "proposed_action": "BLOCAGE_SYSTEME",
        "human_approved": False # Default to false for safety
    }

def execution_agent(state: SystemState):
    """This node only runs AFTER the human has intervened."""
    if state.get("human_approved"):
        print(f"⚙️ Agent Exécuteur: Action validée par l'humain. Exécution de -> {state.get('proposed_action')}")
        return {"messages": ["Action exécutée avec succès."]}
    else:
        print("⚙️ Agent Exécuteur: Action REJETÉE par l'humain. Annulation.")
        return {"messages": ["Action annulée."]}

# ==========================================
# 3. BUILD & COMPILE THE GRAPH
# ==========================================
builder = StateGraph(SystemState)

builder.add_node("planner", planner_agent)
builder.add_node("executor", execution_agent)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", END)

# Initialize memory (crucial for pausing the graph)
memory = MemorySaver()

# COMPILE: We tell LangGraph to PAUSE right before the 'executor' node
graph = builder.compile(checkpointer=memory, interrupt_before=["executor"])

# ==========================================
# 4. RUN THE TEST (Simulation)
# ==========================================
if __name__ == "__main__":
    # A thread ID tracks the specific session/user in memory
    config = {"configurable": {"thread_id": "test_session_001"}}

    print("\n--- DÉMARRAGE DU GRAPH (Partie 1) ---")
    initial_state = {"messages": ["Initialisation du diagnostic."]}
    
    # Stream the graph until it hits the interrupt
    for event in graph.stream(initial_state, config):
        pass 
    
    # Check the state after the graph pauses
    snapshot = graph.get_state(config)
    print("\n⏸️ GRAPH EN PAUSE (Human-in-the-loop déclenché).")
    print(f"Prochain nœud en attente : {snapshot.next}")
    
    # --- INTERVENTION HUMAINE ---
    print("\n--- VALIDATION HUMAINE ---")
    user_input = input("Approuvez-vous l'action 'BLOCAGE_SYSTEME' ? (o/n) : ")
    
    if user_input.lower() == 'o':
        # Update the memory state to approved
        graph.update_state(config, {"human_approved": True})
    else:
        # Update the memory state to rejected
        graph.update_state(config, {"human_approved": False})

    print("\n--- REPRISE DU GRAPH (Partie 2) ---")
    # Resume the graph by passing None as the input
    for event in graph.stream(None, config):
        pass
    
    print("\n✅ GRAPH TERMINÉ.")