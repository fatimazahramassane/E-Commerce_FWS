# src/evaluation.py — Évaluation A/B des prompts (critère 5 RETAL)
"""
Compare Prompt A (court) vs Prompt B (structuré + exemples) sur 10 requêtes test.
Métriques : précision intention, extraction order_id, détection needs_human, validité JSON.
Sortie : console comparative + metrics_results.csv
"""
import sys
from pathlib import Path

import pandas as pd

# Racine projet pour imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.nodes import classify_with_prompt  # noqa: E402
from src.prompts import PROMPT_A_CLASSIFIER, PROMPT_B_CLASSIFIER  # noqa: E402
from src.schemas import ClassificationOutput  # noqa: E402

TEST_CSV = PROJECT_ROOT / "data" / "test_dataset.csv"
OUTPUT_CSV = PROJECT_ROOT / "metrics_results.csv"


def compute_needs_human(intent: str, order_id: str | None) -> bool:
    """Règle métier de référence pour needs_human."""
    return intent == "demande_remboursement" and bool(order_id)


def evaluate_prompt(label: str, prompt_template: str, df: pd.DataFrame) -> dict:
    """Évalue un prompt sur tout le jeu de test."""
    rows = []
    intent_ok = 0
    order_ok = 0
    human_ok = 0
    json_valid = 0
    n = len(df)

    for _, row in df.iterrows():
        query = str(row["Query"])
        expected_intent = str(row["Intent"]).strip()
        expected_order = row["Order_ID"]
        if pd.isna(expected_order) or str(expected_order).strip() == "":
            expected_order = None
        else:
            expected_order = str(expected_order).strip().upper()

        expected_human = str(row["Expected_needs_human"]).lower() in ("true", "1", "yes")

        result, raw, valid_json = classify_with_prompt(query, prompt_template)

        pred_intent = result.intent if result else "autre"
        pred_order = result.order_id if result else None
        pred_human = result.needs_human if result else compute_needs_human(pred_intent, pred_order)

        if valid_json:
            json_valid += 1
        if pred_intent == expected_intent:
            intent_ok += 1
        if pred_order == expected_order:
            order_ok += 1
        if pred_human == expected_human:
            human_ok += 1

        rows.append(
            {
                "prompt": label,
                "query": query,
                "expected_intent": expected_intent,
                "pred_intent": pred_intent,
                "intent_match": pred_intent == expected_intent,
                "expected_order_id": expected_order,
                "pred_order_id": pred_order,
                "order_match": pred_order == expected_order,
                "expected_needs_human": expected_human,
                "pred_needs_human": pred_human,
                "human_match": pred_human == expected_human,
                "json_valid": valid_json,
            }
        )

    return {
        "label": label,
        "intent_accuracy": intent_ok / n if n else 0,
        "order_id_accuracy": order_ok / n if n else 0,
        "needs_human_accuracy": human_ok / n if n else 0,
        "json_validity_rate": json_valid / n if n else 0,
        "details": rows,
    }


def main():
    print("=" * 60)
    print("ÉVALUATION PROMPTS — Support Client E-commerce (RETAL)")
    print("=" * 60)

    if not TEST_CSV.exists():
        print(f"Fichier test introuvable: {TEST_CSV}")
        sys.exit(1)

    df = pd.read_csv(TEST_CSV)
    print(f"\nJeu de test : {len(df)} requêtes depuis {TEST_CSV.name}\n")

    metrics_a = evaluate_prompt("Prompt_A", PROMPT_A_CLASSIFIER, df)
    metrics_b = evaluate_prompt("Prompt_B", PROMPT_B_CLASSIFIER, df)

    summary = pd.DataFrame(
        [
            {
                "prompt": metrics_a["label"],
                "intent_accuracy": metrics_a["intent_accuracy"],
                "order_id_accuracy": metrics_a["order_id_accuracy"],
                "needs_human_accuracy": metrics_a["needs_human_accuracy"],
                "json_validity_rate": metrics_a["json_validity_rate"],
            },
            {
                "prompt": metrics_b["label"],
                "intent_accuracy": metrics_b["intent_accuracy"],
                "order_id_accuracy": metrics_b["order_id_accuracy"],
                "needs_human_accuracy": metrics_b["needs_human_accuracy"],
                "json_validity_rate": metrics_b["json_validity_rate"],
            },
        ]
    )

    print("\n--- RÉSULTATS COMPARATIFS ---\n")
    for col in summary.columns:
        if col != "prompt" and summary[col].dtype in ("float64", "float32"):
            summary[col] = summary[col].apply(lambda x: f"{x:.1%}")
    print(summary.to_string(index=False))

    # Détail par requête
    all_details = metrics_a["details"] + metrics_b["details"]
    details_df = pd.DataFrame(all_details)
    details_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nDétails sauvegardés : {OUTPUT_CSV}")

    # Recommandation
    winner = "B" if metrics_b["intent_accuracy"] >= metrics_a["intent_accuracy"] else "A"
    print(f"\n→ Prompt recommandé pour production : Prompt_{winner}")
    print("=" * 60)


if __name__ == "__main__":
    main()
