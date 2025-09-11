import os
from dotenv import load_dotenv
import re, requests
from langchain_groq import ChatGroq

load_dotenv()

LLM = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

PHASE_ONE_CRITERIA = {
    "Goal": [
        "Implementing a Naive Retrieval-Augmented Generation (RAG) application using LangChain",
    ],
    "Required (5 points each, total 25)": [
        "Document ingestion with chunking (YouTube, PDF, web, text, etc.)",
        "Embeddings generated with chosen model (OpenAI, HuggingFace, etc.)",
        "Vectors stored in a database (FAISS, Chroma, Pinecone, Weaviate, etc.)",
        "Retrieval implemented (dense, sparse, or hybrid)",
        "Prompt template + chain/runnable for grounded generation",
        "Interactive Q&A (CLI, Notebook, Web app, or API)",
    ],
    "Optional Stretch Goals (3 bonus points each)": [
        "Reranker integration (e.g., Cohere Rerank, BM25 + dense retrieval)",
        "Support multiple document sets",
        "Summarization mode in addition to Q&A",
        "Cloud deployment (Vercel, Streamlit Cloud, HuggingFace Spaces, Render, etc.)",
    ],
    "Deliverables": [
        "GitHub repository with complete code",
        "README.md including project description, tech stack, setup, sample queries",
        "Optional: Live deployment link",
    ],
    "Evaluation Criteria": [
        "Core functionality – All required components work correctly",
        "Code clarity – Clean, structured, and readable code",
        "Understanding of concepts – Correct use of LangChain & retrieval",
        "Creativity – Interesting data sources or extra features",
        "Deployment – Bonus for hosted working app",
    ]
}

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
                blobs.append((it["path"], txt[:80000]))  # cap large files
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

      # stretch goals
      "reranker": bool(re.search(r"CrossEncoder|CohereRerank|rerank", big)),
      "multidoc": bool(re.search(r"multiple docs|multi_doc|collection", big)),
      "summarization": bool(re.search(r"summarize|map_reduce|stuff_documents", big)),
      "deployment": bool(re.search(r"render|huggingface_hub|streamlit cloud|vercel", big)),
    }
    return checks, big[:120000]

def evaluate_repo(owner_repo: str) -> str:
    url = f"https://api.github.com/repos/{owner_repo}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            return f"❌ Repository '{owner_repo}' not found on GitHub."
        elif resp.status_code != 200:
            return f"⚠️ Could not fetch repository info (status {resp.status_code})."
    except requests.RequestException as e:
        return f"⚠️ Error connecting to GitHub: {e}"

    checks, code_excerpt = heuristic_scan(owner_repo)
    checklist = "\n".join([f"- {k}: {'✅' if v else '❌'}" for k,v in checks.items()])

    criteria_text = ""
    for section, items in PHASE_ONE_CRITERIA.items():
        criteria_text += f"\n### {section}\n"
        for c in items:
            criteria_text += f"- {c}\n"

    prompt = f"""You are a project evaluator for an AI bootcamp.
Repo: {owner_repo}

Phase One criteria:
{criteria_text}

Heuristic scan checklist:
{checklist}

Task:
1. Score the project:
   - Required: Each ✅ = 5 points, total 25.
   - Stretch Goals: Each ✅ = 3 points, total 12 bonus.
   Report as: "You scored X/25, Bonus Y/12".
2. Write specific TODOs for each ❌ item (missing criteria).
3. Keep feedback concise, actionable, and aligned with the goal: building a Naive RAG chatbot with LangChain.

CODE (excerpt):
{code_excerpt[:6000]}
"""

    response = LLM.invoke(prompt).content
    return "Checklist:\n" + checklist + "\n\nEvaluation:\n" + response
