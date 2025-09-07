# rag/reranker.py
from sentence_transformers import CrossEncoder

# Load once
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, docs, top_k=4):
    pairs = [(question, d.page_content) for d in docs]
    scores = RERANKER.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored_docs[:top_k]]
