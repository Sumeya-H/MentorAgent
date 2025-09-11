import os
from dotenv import load_dotenv
from evaluator.repo_eval import evaluate_repo
from langchain_groq import ChatGroq
from rag.prompts import SYSTEM, QA_TEMPLATE, REFLECT_PROMPT
from rag.index import retriever_topk   

load_dotenv()

LLM = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def tool_search_docs(q, vs, k=4):
    docs = retriever_topk(vs, q, k=k)
    return {"type":"search_results", "results": [{"text":d.page_content, "meta":d.metadata} for d in docs]}

def tool_fetch_repo(owner_repo):
    # return README content
    from rag.loaders import load_github_readme
    docs = load_github_readme(owner_repo)
    if not docs:
        return {"type":"repo_not_found"}
    return {"type":"repo_readme", "readme": docs[0].page_content, "meta": docs[0].metadata}

def tool_eval_repo(owner_repo):
    return {"type":"repo_eval", "report": evaluate_repo(owner_repo)}

def tool_summarize(text, max_sentences=5):
    prompt = f"Summarize this into {max_sentences} sentences:\n\n{text}"
    return {"type":"summary", "summary": LLM.invoke([{"role":"user","content":prompt}]).content}
