import os
from dotenv import load_dotenv
import re, requests
from langchain_groq import ChatGroq

load_dotenv()

LLM = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

PHASE_ONE_CRITERIA = [
  "Document ingestion with chunking",
  "Embeddings generated (model noted)",
  "Vector store used (FAISS/Chroma/Pinecone/etc.)",
  "Retrieval implemented (dense/sparse/hybrid)",
  "Prompt template + chain/runnable for generation",
  "User interaction (CLI/Notebook/Web/API)"
]

def fetch_repo_tree(owner_repo: str):
    url = f"https://api.github.com/repos/{owner_repo}/git/trees/HEAD?recursive=1"
    r = requests.get(url); r.raise_for_status()
    return r.json().get("tree", [])

def fetch_raw(owner_repo: str, path: str) -> str:
    url = f"https://raw.githubusercontent.com/{owner_repo}/HEAD/{path}"
    r = requests.get(url)
    return r.text if r.status_code == 200 else ""

def heuristic_scan(owner_repo: str):
    blobs = []
    for it in fetch_repo_tree(owner_repo):
        if it["type"] == "blob" and it["path"].endswith((".py",".md",".ipynb",".js",".ts",".txt")):
            txt = fetch_raw(owner_repo, it["path"])
            if txt:
                blobs.append((it["path"], txt[:80000]))  # cap
    big = "\n\n".join([f"## {p}\n{t}" for p,t in blobs])

    checks = {
      "ingestion": bool(re.search(r"PDFLoader|PyPDF|BeautifulSoup|WebBaseLoader|NotebookLoader", big)),
      "chunking":  bool(re.search(r"TextSplitter|RecursiveCharacterTextSplitter|chunk", big)),
      "embedding": bool(re.search(r"sentence-transformers|OpenAIEmbeddings|HuggingFaceEmbeddings", big)),
      "vectordb":  bool(re.search(r"FAISS|Chroma|Pinecone|Weaviate", big)),
      "retrieval": bool(re.search(r"similarity_search|as_retriever|BM25", big)),
      "prompt":    bool(re.search(r"PromptTemplate|Runnable|Chain", big)),
      "ui":        bool(re.search(r"streamlit|gradio|fastapi|flask|cli", big)),
      "streamlit": bool(re.search(r"import streamlit as st", big)),
    }
    return checks, big[:120000]

def evaluate_repo(owner_repo: str) -> str:
    checks, code_excerpt = heuristic_scan(owner_repo)
    checklist = "\n".join([f"- {k}: {'✅' if v else '❌'}" for k,v in checks.items()])
    criteria_text = "\n".join([f"- {c}" for c in PHASE_ONE_CRITERIA])
    prompt = f"""You are a concise reviewer.
Repo: {owner_repo}

Phase One criteria:
{criteria_text}

Heuristic scan checklist:
{checklist}

Based on the code excerpt, write specific TODOs to meet missing criteria. Be concrete and brief.
CODE:
{code_excerpt[:6000]}
"""
    return "Checklist:\n" + checklist + "\n\nFeedback:\n" + LLM.invoke(prompt).content
