import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# ✅ FIX: Résolution du chemin relatif au fichier ingest.py, pas au répertoire
# courant d'exécution. L'ancien chemin "data/politique_remboursement.txt"
# échouait sauf si on lançait le script depuis un dossier précis.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(BASE_DIR, "politique_remboursement.txt")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


def ingest_documents(policy_path: str = POLICY_PATH, chroma_dir: str = CHROMA_DIR):
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Fichier introuvable : {policy_path}")

    loader = TextLoader(policy_path, encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ✅ FIX: Suppression de vectorstore.persist() — déprécié depuis
    # langchain-chroma 0.1.x. Chroma persiste automatiquement dès que
    # persist_directory est fourni dans from_documents().
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=chroma_dir,
    )
    print(f"✅ {len(chunks)} chunks indexés dans ChromaDB ({chroma_dir}).")


if __name__ == "__main__":
    ingest_documents()