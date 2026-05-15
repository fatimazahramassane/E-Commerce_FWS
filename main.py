import asyncio
from src.graph.workflow import graph
from langgraph.types import Command

async def handle_stream(stream):
    """Affiche les événements et détecte une interruption."""
    async for event in stream:
        for node_name, output in event.items():
            print(f"[{node_name}] -> {output}")

        # Détection interruption
        if "__interrupt__" in event:
            interrupt_info = event["__interrupt__"]
            print("\n🛑 INTERRUPTION - Validation humaine demandée :")
            for item in interrupt_info:
                print(item)
            return True  # Indique qu'il faut reprendre
    return False  # Pas d'interruption

async def run():
    config = {"configurable": {"thread_id": "test1"}}
    initial_state = {
        "query": "Je n'ai pas reçu ma commande #12345, je veux être remboursé",
        "intent": None,
        "order_id": None,
        "retrieved_context": None,
        "proposed_response": None,
        "proposed_action": None,
        "requires_human": False,
        "final_response": None,
        "approved": None
    }

    print(">>> Début du workflow...")
    stream = graph.astream(initial_state, config)
    interrupted = await handle_stream(stream)

    if interrupted:
        reponse = input("Approuver le remboursement ? (oui/non) : ").strip().lower()
        approval = reponse == "oui"
        print(">>> Reprise du workflow...")
        resume_stream = graph.astream(Command(resume={"approved": approval}), config)
        await handle_stream(resume_stream)

if __name__ == "__main__":
    asyncio.run(run())