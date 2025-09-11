import os
import json, re
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from rag.prompts import SYSTEM, QA_TEMPLATE, REFLECT_PROMPT, PLANNER_PROMPT
from rag.index import load_chroma, retriever_topk
from evaluator.repo_eval import evaluate_repo
from rag.retrievers import hybrid_retrieve
from rag.reranker import rerank
from agent.tools import tool_search_docs, tool_fetch_repo, tool_eval_repo, tool_summarize

load_dotenv()

# Singletons
LLM = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
VERIFIER = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
PLANNER = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
VS = None
def _vs():
    global VS
    if VS is None:
        VS = load_chroma("vectorstore")
    return VS

def safe_parse_plan(resp: str):
    try:
        # Try direct parse first
        return json.loads(resp)
    except Exception:
        # Extract first JSON block with regex
        match = re.search(r"\[.*\]", resp, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    # Fallback
    return [{"tool": "final_answer", "args": {"note": "Raw plan text"}}]

# -------- Router --------
def router_node(state):
    print("<<<<<<<<<<<<<<<<<<<---- Router node activated.--->>>>>>>>>>>>>>>>>>>")
    print("Router input state:", state)
    q = state["question"].strip().lower()

    if "reset memory" in q or "clear history" in q:
        return {"route": "reset"}
    
    route = "qa"
    if "evaluate repo" in q or "review repo" in q or "check repo" in q or state.get("repo"):
        route = "repo_eval"
    elif "show me" in q and ("project" in q or "repo" in q or "pinecone" in q or "streamlit" in q):
        route = "search"
    # Planner (preparation requests)
    elif (
        "plan" in q
        or "prepare" in q
        or "help me prepare for" in q
        or "prep" in q
        or "roadmap" in q
    ):
        route = "planner"
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

    # Use last 3 turns of memory
    history = "\n".join([f"Q: {m['q']}\nA: {m['a']}" for m in state.get("memory", [])[-3:]])
    full_context = (history + "\n\n" if history else "") + (state.get("context") or "NO CONTEXT")

    prompt = ChatPromptTemplate.from_template(QA_TEMPLATE).format(
        question=state["question"],
        context=full_context
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
    
    # History (last 3 turns)
    history = "\n".join([f"Q: {m['q']}\nA: {m['a']}" for m in state.get("memory", [])[-3:]])
    full_context = (history + "\n\n" if history else "") + (state.get("context") or "NO CONTEXT")
    # Build verifier prompt
    prompt = f"""Question: {state['question']}
Context: {full_context}
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

    # increment reflect retries if failed
    retries = state.get("reflect_retries", 0)
    if not j.get("verified", False):
        retries += 1

    return {
        "answer": final_answer,
        "verified": bool(j.get("verified", False)),
        "memory": mem,
        "issues": j.get("issues", []),
        "reflect_retries": retries
    }

# -------- Reset memory (QA) --------
def reset_node(state):
    print("<<<<<<<<<<<<<<<<<<<---- Reset node activated.--->>>>>>>>>>>>>>>>>>>")
    print("Reset input state:", state)
    return {"answer": "✅ Memory has been cleared.", "memory": []}

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

# -------- Planer (Agent) --------
def planner_node(state):
    print("<<<<<<<<<<<<<<<<<<<---- Planner node activated.--->>>>>>>>>>>>>>>>>>>")
    print("Planner input state:", state)
    # Compose prompt with goal and available context
    prompt = PLANNER_PROMPT + f"\nUser goal: {state['question']}\nContext: {state.get('context','')}\n"
    resp = PLANNER.invoke([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}]).content
    try:
        #plan = json.loads(resp)
        plan = safe_parse_plan(resp)
        answer = "Here’s your step-by-step plan:\n" + json.dumps(plan, indent=2) 
    except Exception:
        plan = [{"tool":"final_answer","args":{"note":"Raw plan text"}}]
        answer = resp   # use the raw text from LLM
    #return {"plan": plan, "answer": answer, "plan_retries": retries}
    #try:
        #plan = json.loads(resp)
    #except Exception:
        # Fallback: single final_answer action
        #plan = [{"tool":"final_answer","args":{"note":"Could not plan; fallback answer."}}]
    retries = state.get("plan_retries", 0)
    return {"plan": plan, "answer": answer, "plan_retries": retries}

# -------- Action Executor (Agent) --------
def action_executor_node(state):
    print("<<<<<<<<<<<<<<<<<<<---- Action Executor node activated.--->>>>>>>>>>>>>>>>>>>")
    print("Action Executor input state:", state)
    plan = state.get("plan", [])
    base_answer = state.get("answer", "Plan results:\n")
    results = []
    for action in plan:
        tool = action.get("tool")
        args = action.get("args", {})
        if tool == "search_docs":
            results.append(tool_search_docs(args.get("q",""),_vs(), k=4))
        elif tool == "fetch_repo":
            results.append(tool_fetch_repo(args.get("repo","")))
        #elif tool == "eval_repo":
            #results.append(tool_eval_repo(args.get("repo","")))
        elif tool == "summarize":
            results.append(tool_summarize(args.get("text",""), max_sentences=args.get("max",5)))
        elif tool == "final_answer":
            # Build final answer from results and break
            final_text = base_answer + "\n\n"+ "Plan results:\n" + "\n".join([str(r) for r in results])
            #final_text = "Plan results:\n"
            for r in results:
                if r["type"] == "search_results":
                    final_text += "\n".join([s["text"][:300] for s in r["results"][:2]])
                elif r["type"] == "repo_readme":
                    final_text += f"\nREADME excerpt:\n{r['readme'][:400]}"
                elif r["type"] == "repo_eval":
                    final_text += f"\nRepo eval:\n{r['report']}"
                elif r["type"] == "summary":
                    final_text += f"\nSummary:\n{r['summary']}"
            return {"answer": final_text}
    # If no final_answer encountered:
    return {"answer": "Plan executed. No final synthesized answer produced."}