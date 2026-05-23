# llm.py — Client Groq (Llama-3.3-70B)
import json
import os
import re
from typing import Any, Optional

from langchain_groq import ChatGroq


def get_groq_llm(temperature: float = 0.1) -> ChatGroq:
    """Instancie le LLM Groq avec la clé depuis l'environnement."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY manquante. Définissez-la dans .env ou les variables d'environnement."
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=api_key,
    )


def extract_json_from_text(text: str) -> Optional[dict[str, Any]]:
    """Extrait un objet JSON depuis une réponse LLM (markdown ou brut)."""
    if not text:
        return None
    text = text.strip()
    # Bloc ```json ... ```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    # Objet JSON direct
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def invoke_llm_safe(llm: ChatGroq, prompt: str) -> tuple[str, Optional[str]]:
    """Appel LLM avec gestion d'erreurs."""
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return str(content), None
    except Exception as exc:
        return "", f"Erreur LLM: {exc}"
