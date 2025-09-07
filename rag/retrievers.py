# rag/retrievers.py
from langchain_community.retrievers import BM25Retriever
from rag.index import load_chroma, retriever_topk

def hybrid_retrieve(question, vs, k=4):
    """Combine dense and sparse retrieval, return merged results."""
    # Dense retrieval (from Chroma retriever)
    dense_docs = retriever_topk(vs, question, k=k)

    # Get all docs stored in Chroma (flat texts)
    vs_data = vs.get()
    # vs_data["documents"] is a list of lists (1 list per vector/embedding)
    all_texts = [doc for docs in vs_data["documents"] for doc in docs]

    # Sparse retrieval (BM25 over raw texts)
    bm25 = BM25Retriever.from_texts(all_texts)
    sparse_docs = bm25.get_relevant_documents(question)[:k]

    # Merge results
    docs = dense_docs + sparse_docs

    # Deduplicate by text content
    seen = set()
    unique = []
    for d in docs:
        text = d.page_content if hasattr(d, "page_content") else str(d)
        if text not in seen:
            seen.add(text)
            # Ensure everything is a Document object
            if not hasattr(d, "page_content"):
                from langchain_core.documents import Document
                d = Document(page_content=text, metadata={"source": "bm25"})
            unique.append(d)

    return unique[:k]
