import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain.schema import Document

from rag.loaders import load_pdf, load_markdown, load_ipynb, load_youtube_transcript, load_github_readme
from rag.chunking import split_and_tag
from rag.index import build_chroma, load_chroma
from graph.build_graph import compile_graph

load_dotenv()
st.set_page_config(page_title="Mentor Agent (NSK.AI)", layout="wide")
st.title("🤖 Mentor Agent — NSK.AI")

persist_dir = "vectorstore"
graph = compile_graph()

# ------ Sidebar: Build Index ------
with st.sidebar:
    st.header("Index Builder")
    uploaded = st.file_uploader("Upload PDFs/MD/IPYNB", type=["pdf","md","ipynb"], accept_multiple_files=True)
    yt = st.text_input("YouTube URL (optional)")
    gh = st.text_input("GitHub repo (owner/name) to index README (optional)")
    if st.button("Build / Update Index"):
        docs = []
        if uploaded:
            Path("data").mkdir(exist_ok=True)
            for f in uploaded:
                p = Path("data")/f.name
                p.write_bytes(f.read())
                if f.name.endswith(".pdf"): docs += load_pdf(p)
                elif f.name.endswith(".md"): docs += load_markdown(p)
                elif f.name.endswith(".ipynb"): docs += load_ipynb(p)
        if yt: docs += load_youtube_transcript(yt)
        if gh: docs += load_github_readme(gh)
        if not docs:
            st.warning("No documents to index.")
        else:
            chunks = split_and_tag(docs)
            build_chroma(chunks, persist_dir=persist_dir)
            st.success(f"Indexed {len(chunks)} chunks.")

tab1, tab2, tab3 = st.tabs(["Chat (Q&A)", "Repo Evaluator", "Knowledge Search"])

with tab1:
    st.subheader("Ask about bootcamp materials")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Clear Chat Button ---
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []   # clear UI chat
        if "memory" in st.session_state:     # if you keep agent memory in state
            st.session_state.memory = []     # clear memory too
        st.rerun()  

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:  ** {msg['content']}")
        else:
            st.markdown(f"**🤖 Mentor Agent:  ** {msg['content']}")

    # --- Input handling ---
    def handle_input():
        q = st.session_state.chat_input
        if q:
            # Save user message
            st.session_state.chat_history.append({"role": "user", "content": q})

            # Call graph
            if not (Path(persist_dir).exists() and any(Path(persist_dir).iterdir())):
                answer = "Please build the index first (sidebar)."
            else:
                out = graph.invoke({"question": q})
                answer = out.get("answer", "(no answer)")

            # Save assistant message
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Clear input after sending
            st.session_state.chat_input = ""

    # Chat input with on_change
    st.text_input(
        "Type your question here...",
        key="chat_input",
        on_change=handle_input
    )

with tab2:
    st.subheader("Evaluate a Phase One repo")
    repo = st.text_input("Public GitHub repo (owner/name)", placeholder="user/project", key="repoinput")
    if st.button("Evaluate Repo"):
        q = f"evaluate repo {repo}" if repo else "evaluate repo"
        out = graph.invoke({"question": q, "repo": repo})
        st.markdown("### Report")
        st.write(out.get("answer","(no answer)"))

with tab3:
    st.subheader("Semantic search over indexed repos/materials")
    q = st.text_input("Search query (e.g., Phase One project with RAG + Pinecone)", key="search")
    if st.button("Search"):
        if not (Path(persist_dir).exists() and any(Path(persist_dir).iterdir())):
            st.info("Please build the index first (sidebar).")
        else:
            out = graph.invoke({"question": q})
            st.markdown("### Results")
            st.write(out.get("answer","(no results)"))
