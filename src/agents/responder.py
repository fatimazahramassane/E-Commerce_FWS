from src.graph.state import SupportState

def responder_node(state: SupportState) -> dict:
    # Simule l'envoi. En réalité, on pourrait appeler une API email/chat.
    print("--- RÉPONSE ENVOYÉE AU CLIENT ---")
    print(state.get("final_response", "Réponse vide."))
    print("--------------------------------")
    return {"final_response": state.get("final_response")}