# schemas.py — Validation Pydantic stricte (classifier)
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


VALID_INTENTS = (
    "suivi_commande",
    "demande_remboursement",
    "reclamation",
    "autre",
)


class ClassificationOutput(BaseModel):
    """Schéma JSON attendu du nœud classifier — évite les crashes de parsing."""

    intent: Literal[
        "suivi_commande",
        "demande_remboursement",
        "reclamation",
        "autre",
    ] = Field(description="Intention détectée du client")
    order_id: Optional[str] = Field(
        default=None,
        description="Numéro de commande extrait (format ORD-XXXXX) ou null",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    needs_human: bool = Field(
        default=False,
        description="True si validation financière humaine requise",
    )
    reasoning: str = Field(default="", description="Brève justification")

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None or str(v).strip() in ("", "null", "none", "None"):
            return None
        cleaned = str(v).strip().upper()
        if cleaned.startswith("ORD-"):
            return cleaned
        # Tentative d'extraction si format partiel
        if cleaned.isdigit() or (len(cleaned) > 3 and "-" in cleaned):
            if not cleaned.startswith("ORD-"):
                return f"ORD-{cleaned.replace('ORD', '').strip('-')}"
        return cleaned if cleaned else None

    @field_validator("needs_human", mode="before")
    @classmethod
    def coerce_needs_human(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "oui")
        return bool(v)
