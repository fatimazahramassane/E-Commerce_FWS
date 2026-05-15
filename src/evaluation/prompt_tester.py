import re
import os
import csv
import json
import logging
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

PROMPT_A = """Tu es un agent de support client. Analyse la demande et réponds au format JSON : {"intent": "...", "order_id": "..."}
Les intentions possibles sont : suivi_commande, demande_remboursement, réclamation, autre."""

PROMPT_B = """Tu es un agent de support client expérimenté. Ta tâche est d'analyser le message d'un client et d'en extraire deux informations :
1. L'intention principale : "suivi_commande" (veut savoir où en est sa commande), "demande_remboursement" (exige un remboursement), "réclamation" (se plaint d'un produit/service), ou "autre" (tout le reste).
2. Le numéro de commande si mentionné (par exemple #12345), sinon null.

Exemples :
- "Je n'ai pas reçu ma commande #12345, je veux être remboursé" → {"intent": "demande_remboursement", "order_id": "12345"}
- "Où est ma commande #98765 ?" → {"intent": "suivi_commande", "order_id": "98765"}
- "Le produit est cassé, que faire ?" → {"intent": "réclamation", "order_id": null}

IMPORTANT : réponds UNIQUEMENT au format JSON, sans texte supplémentaire ni balises markdown."""


def _parse_json_safe(text: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(clean)


def _normalize_order_id(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in ("none", "", "nan"):
        return None
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def evaluate_prompt(prompt_version: str, dataset_path: str):
    total = 0
    correct_intent = 0
    correct_order_id = 0
    correct_needs_human = 0
    results = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            query = row["query"]
            expected_intent = row["expected_intent"].strip()
            expected_order_id = _normalize_order_id(row.get("expected_order_id"))
            expected_needs = row.get("expected_needs_human", "False").strip() == "True"

            messages = [SystemMessage(content=prompt_version), HumanMessage(content=query)]
            try:
                response = llm.invoke(messages)
                parsed = _parse_json_safe(response.content)
            except Exception as e:
                logger.warning(f"Parse error: {e}")
                parsed = {"intent": "autre", "order_id": None}

            pred_intent = parsed.get("intent", "autre").strip()
            pred_order_id = _normalize_order_id(parsed.get("order_id"))
            pred_needs = pred_intent == "demande_remboursement"

            intent_ok = pred_intent == expected_intent
            order_ok = pred_order_id == expected_order_id
            needs_ok = pred_needs == expected_needs

            if intent_ok: correct_intent += 1
            if order_ok: correct_order_id += 1
            if needs_ok: correct_needs_human += 1

            results.append({
                "query": query,
                "expected_intent": expected_intent, "predicted_intent": pred_intent, "intent_ok": intent_ok,
                "expected_order_id": expected_order_id, "predicted_order_id": pred_order_id, "order_ok": order_ok,
                "expected_needs_human": expected_needs, "predicted_needs_human": pred_needs, "needs_ok": needs_ok,
            })

    acc_intent = (correct_intent / total * 100) if total else 0
    acc_order  = (correct_order_id / total * 100) if total else 0
    acc_needs  = (correct_needs_human / total * 100) if total else 0
    return results, acc_intent, acc_order, acc_needs


def print_results(label, results, acc_intent, acc_order, acc_needs):
    print(f"\n=== {label} ===")
    print(f"  Précision intention     : {acc_intent:.1f}%")
    print(f"  Précision order_id      : {acc_order:.1f}%")
    print(f"  Précision needs_human   : {acc_needs:.1f}%")
    print("\n  Détail des erreurs :")
    for r in results:
        if not r["intent_ok"] or not r["order_ok"]:
            tag = "[INTENT✗]" if not r["intent_ok"] else ""
            tag += "[ORDER✗]" if not r["order_ok"] else ""
            print(f"  {tag} {r['query'][:60]}")
            if not r["intent_ok"]:
                print(f"    intent: expected={r['expected_intent']} got={r['predicted_intent']}")
            if not r["order_ok"]:
                print(f"    order: expected={r['expected_order_id']} got={r['predicted_order_id']}")


if __name__ == "__main__":
    # ✅ FIX: Chemin résolu par rapport à l'emplacement du script, plus par
    # rapport au répertoire courant d'exécution. L'ancien chemin codé en dur
    # "src/evaluation/test_dataset.csv" levait FileNotFoundError selon le CWD.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset = os.path.join(BASE_DIR, "test_dataset.csv")

    results_a, acc_a_i, acc_a_o, acc_a_n = evaluate_prompt(PROMPT_A, dataset)
    print_results("Prompt A (court)", results_a, acc_a_i, acc_a_o, acc_a_n)

    results_b, acc_b_i, acc_b_o, acc_b_n = evaluate_prompt(PROMPT_B, dataset)
    print_results("Prompt B (détaillé)", results_b, acc_b_i, acc_b_o, acc_b_n)

    print("\n--- Comparaison ---")
    print(f"Prompt A : intent={acc_a_i:.1f}%  order={acc_a_o:.1f}%  needs={acc_a_n:.1f}%")
    print(f"Prompt B : intent={acc_b_i:.1f}%  order={acc_b_o:.1f}%  needs={acc_b_n:.1f}%")
    winner = "B (détaillé)" if acc_b_i >= acc_a_i else "A (court)"
    print(f"→ Meilleur prompt sur l'intention : Prompt {winner}")