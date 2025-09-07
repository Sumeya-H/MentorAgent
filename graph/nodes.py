import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from rag.prompts import SYSTEM, QA_TEMPLATE, REFLECT_PROMPT
from rag.index import load_chroma, retriever_topk
from evaluator.repo_eval import evaluate_repo
from rag.retrievers import hybrid_retrieve
from rag.reranker import rerank

load_dotenv()

# Singletons
LLM = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
VERIFIER = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
VS = None
def _vs():
    global VS
    if VS is None:
        VS = load_chroma("vectorstore")
    return VS

# -------- Router --------
def router_node(state):
    print("<<<<<<<<<<<<<<<<<<<---- Router node activated.--->>>>>>>>>>>>>>>>>>>")
    print("Router input state:", state)
    q = state["question"].strip().lower()
    route = "qa"
    if "evaluate repo" in q or "review repo" in q or "check repo" in q or state.get("repo"):
        route = "repo_eval"
    elif "show me" in q and ("project" in q or "repo" in q or "pinecone" in q or "streamlit" in q):
        route = "search"
    return {"route": route}

# -------- Retrieve (for QA) --------
def retrieve_node(state):
    print("<<<<<<<<<<<<<<<<<<<---- Retrieve node activated.--->>>>>>>>>>>>>>>>>>>")
    print("Retrieve input state:", state)
    docs = hybrid_retrieve(state["question"], _vs(), k=8)   # get more
    docs = rerank(state["question"], docs, top_k=4)         # keep best
    
    ctx = "\n\n".join([f"[{i}] {d.page_content}" for i,d in enumerate(docs)])
    refs = [d.metadata | {"id": i} for i,d in enumerate(docs)]  # track index
    
    return {"context": ctx, "refs": refs}

# -------- Generate (QA) --------
def generate_node(state):
    print("<<<<<<<<<<<<<<<<<<<---- Generate node activated.--->>>>>>>>>>>>>>>>>>>")
    print("Generate input state:", state)
    prompt = ChatPromptTemplate.from_template(QA_TEMPLATE).format(
        question=state["question"],
        context=state["context"] or "NO CONTEXT"
    )
    msgs = [{"role":"system","content":SYSTEM},{"role":"user","content":prompt}]
    out = LLM.invoke(msgs).content

    # Save candidate answer into state
    state["candidate_answer"] = out 

    # Inline citations: replace doc markers [i]
    cited_answer = out
    for ref in state.get("refs", []):
        cited_answer = cited_answer.replace(f"[{ref['id']}]", f"[{ref['id']}]")

    # Tail references
    tail = ["\nReferences:"]
    for r in state.get("refs", []):
        src = r.get("source_file") or r.get("repo") or r.get("video_id") or r.get("path") or r.get("source","?")
        tail.append(f"[{r['id']}] {src} | {r.get('reference', r.get('chapter','?'))}")
    
    final_answer = cited_answer + "\n" + "\n".join(tail)
    return {"answer": final_answer, "candidate_answer": out}

# -------- Reflect (QA) --------
def reflect_node(state):
    print("<<<<<<<<<<<<<<<<<<<---- Reflect node activated.--->>>>>>>>>>>>>>>>>>>")
    print("Reflect input state:", state)

    # Use candidate_answer from generate_node for reflection
    candidate = state.get("candidate_answer", state.get("answer", ""))
    # Build verifier prompt
    prompt = f"""Question: {state['question']}
Context: {state.get('context','')}
Candidate answer: {candidate}
{REFLECT_PROMPT}
"""
    resp = VERIFIER.invoke(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    ).content.strip()

    # Parse JSON
    import json
    try:
        j = json.loads(resp)
    except Exception:
        j = {
            "verified": False,
            "answer": candidate,
            "issues": ["Verifier failed to return JSON; manual check needed."],
        }

    # Tail references (same as generate_node)
    tail = ["\nReferences:"]
    for r in state.get("refs", []):
        src = (
            r.get("source_file")
            or r.get("repo")
            or r.get("video_id")
            or r.get("path")
            or r.get("source", "?")
        )
        tail.append(f"[{r['id']}] {src} | {r.get('reference', r.get('chapter','?'))}")

    # Final answer = verified answer + references
    final_answer = j.get("answer", candidate) + "\n" + "\n".join(tail)

    # Update memory
    mem = state.get("memory", [])
    mem.append(
        {"q": state["question"], "a": j.get("answer", candidate), "refs": str(state.get("refs", []))}
    )

    return {
        "answer": final_answer,
        "verified": bool(j.get("verified", False)),
        "memory": mem,
        "issues": j.get("issues", []),
    }
# -------- Repo Evaluator --------
def repo_eval_node(state):
    print("<<<<<<<<<<<<<<<<<<<---- Repo Eval node activated.--->>>>>>>>>>>>>>>>>>>")
    print("Repo Eval input state:", state)
    repo = state.get("repo")
    if not repo:
        # try to parse repo from the question (very light heuristic)
        import re
        m = re.search(r"([\w\-]+\/[\w\.\-]+)", state["question"])
        repo = m.group(1) if m else None
    if not repo:
        return {"answer": "Please provide a public GitHub repo as owner/name."}
    report = evaluate_repo(repo)
    return {"answer": report}

# -------- Knowledge Search (repos) --------
def search_node(state):
    print("<<<<<<<<<<<<<<<<<<<---- Search node activated.--->>>>>>>>>>>>>>>>>>>")
    print("Search input state:", state)
    # reuse vector store with repo READMEs indexed
    docs = retriever_topk(_vs(), state["question"], k=5)
    lines = []
    for d in docs:
        repo = d.metadata.get("repo") or d.metadata.get("source_file","")
        snippet = d.page_content[:350].replace("\n"," ")
        lines.append(f"- {repo} — {snippet}...")
    ans = "Here are relevant repositories/snippets:\n" + "\n".join(lines)
    return {"answer": ans}
