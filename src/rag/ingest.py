# src/rag/ingest.py — Ingestion RAG : chunking, embeddings, ChromaDB persistant
"""
Ingestion de policy_remboursement.txt :
- Chunks de 300 caractères, overlap 50
- Embeddings HuggingFace all-MiniLM-L6-v2
- Store ChromaDB persistant dans ./chroma_db
"""
import logging
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins relatifs à la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "data" / "policy_remboursement.txt"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "policy_remboursement"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe le texte en chunks avec chevauchement."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """Embeddings sentence-transformers all-MiniLM-L6-v2."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_policy_text() -> str:
    """Charge le fichier politique de remboursement."""
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Fichier politique introuvable: {POLICY_PATH}")
    return POLICY_PATH.read_text(encoding="utf-8")


def build_documents() -> list[Document]:
    """Construit les documents LangChain à partir des chunks."""
    raw = load_policy_text()
    chunks = chunk_text(raw)
    return [
        Document(
            page_content=chunk,
            metadata={"source": "policy_remboursement.txt", "chunk_id": i},
        )
        for i, chunk in enumerate(chunks)
    ]


def ingest_policy(force: bool = False) -> Chroma:
    """
    Ingère la politique dans ChromaDB (persistant).
    Si force=False et collection existante, réutilise le store.
    """
    embeddings = get_embeddings()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if not force and CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        logger.info("Chargement ChromaDB existant depuis %s", CHROMA_DIR)
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    documents = build_documents()
    logger.info("Ingestion de %d chunks dans ChromaDB", len(documents))
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


def get_vectorstore() -> Chroma:
    """Retourne le vectorstore (ingère si nécessaire)."""
    return ingest_policy(force=False)


def retrieve_context(query: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
    """
    Retrieval top_k=3 avec fallback si aucun chunk.
    Retourne (contexte concaténé, liste des chunks).
    """
    fallback_msg = (
        "Aucun extrait de politique trouvé. Appliquer les délais standards : "
        "14 jours de rétractation, remboursement sous 7 jours ouvrés après retour."
    )
    try:
        vs = get_vectorstore()
        results = vs.similarity_search(query, k=top_k)
        if not results:
            logger.warning("RAG fallback: aucun chunk pour la requête: %s", query[:80])
            return fallback_msg, []
        chunks = [doc.page_content for doc in results]
        context = "\n---\n".join(chunks)
        return context, chunks
    except Exception as exc:
        logger.error("Erreur retrieval RAG: %s", exc)
        return fallback_msg + f" (log: {exc})", []


if __name__ == "__main__":
    """Exécution directe : python -m src.rag.ingest"""
    os.chdir(PROJECT_ROOT)
    store = ingest_policy(force=True)
    ctx, chunks = retrieve_context("remboursement commande 500 euros")
    print(f"Chunks récupérés: {len(chunks)}")
    print(ctx[:500])
