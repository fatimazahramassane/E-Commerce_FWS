from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

def retrieve_context(query: str, k: int = 3) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    return context