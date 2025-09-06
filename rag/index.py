from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List

EMB = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_chroma(docs: List[Document], persist_dir: str = "vectorstore"):
    # Try to load existing vectorstore, else create new
    try:
        vs = Chroma(embedding_function=EMB, persist_directory=persist_dir)
        vs.add_documents(docs)
    except Exception:
        # If loading fails (e.g., directory doesn't exist), create new
        vs = Chroma.from_documents(docs, embedding=EMB, persist_directory=persist_dir)
        vs.persist()
    return vs

def load_chroma(persist_dir: str = "vectorstore"):
    return Chroma(embedding_function=EMB, persist_directory=persist_dir)

def retriever_topk(vs, query: str, k: int = 4):
    return vs.similarity_search(query, k=k)
