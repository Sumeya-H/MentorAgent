# 🤖 MentorAgent — AI-Powered Bootcamp Assistant

MentorAgent is a modular, intelligent assistant built to support learners and developers during the NSK.AI Bootcamp. It combines Retrieval-Augmented Generation (RAG), semantic search, and graph-based reasoning to deliver personalized mentorship, code evaluation, and knowledge discovery—all from your documents, repos, and transcripts.

---

## 🧠 Architecture Overview

MentorAgent is powered by a multi-node LangGraph pipeline that routes user queries through a series of intelligent steps:

```
User Query
   ↓
[Router Node]
   |
   |--(route == "qa")------> [Retrieve Node] ---> [Generate Node] ---> [Reflect Node] ---> [END]
   |
   |--(route == "repo_eval")--> [Repo Eval Node] ---> [END]
   |
   |--(route == "search")------> [Search Node] ---> [END]


[Router Node] → Determines intent (Q&A, Repo Eval, Search)
[Retrieve Node] → Hybrid dense + sparse retrieval
[Generate Node] → LLM-based answer generation with citations
[Reflect Node] → Verifies answer quality using a second LLM
[Repo Eval Node] → Specialized logic for repo scoring and feedback
[Search Node] → Specialized logic for semantic search
```

Each node operates on a shared `AgentState`, enabling memory, context, and traceability across turns.

---

## 🚀 Features

- 📄 **Multi-format Ingestion**: PDFs, Markdown, Jupyter Notebooks, YouTube transcripts, GitHub READMEs
- 🔍 **Hybrid Retrieval**: Combines dense vector search with BM25 keyword matching
- 🧠 **Cross-Encoder Reranking**: Prioritizes the most relevant chunks using semantic scoring
- 🧾 **Inline Citations**: Answers include traceable references to source documents
- 🧪 **Verifier Node**: Uses a second LLM to reflect and validate generated answers
- 📊 **Repo Evaluator**: Automatically scores GitHub projects against bootcamp criteria
- 🔎 **Semantic Search**: Explore indexed repos and materials with natural language queries
- 🧼 **Modular Design**: Clean separation of loaders, retrievers, indexers, and graph logic

---

## 🛠️ Tech Stack

| Layer            | Technology Used                           |
|------------------|--------------------------------------------|
| Language         | Python                                     |
| Framework        | LangChain, LangGraph, Streamlit            |
| Embeddings       | HuggingFace `all-MiniLM-L6-v2`             |
| Vector Store     | Chroma                                     |
| Reranker         | SentenceTransformers CrossEncoder          |
| LLMs             | Groq `llama-3.1-8b-instant`                |
| UI               | Streamlit                                  |
| Repo Analysis    | GitHub API + Heuristic Regex Scanner       |

---

## 📁 Project Structure

```
MentorAgent/
├── app.py                 # Streamlit frontend
├── graph/                 # LangGraph pipeline and nodes
│   ├── build_graph.py
│   ├── node.py
│   └── state.py
├── rag/                   # RAG components
│   ├── loaders.py
│   ├── retrievers.py
│   ├── reranker.py
│   ├── chunking.py
│   └── index.py
├── evaluator/             # GitHub repo evaluator
│   └── repo_eval.py
├── vectorstore/           # Persisted Chroma DB
├── data/                  # Uploaded documents
└── requirements.txt       # Dependencies
```

---

## ⚙️ Setup & Usage

### 1. Clone and Install
```bash
git clone https://github.com/Sumeya-H/MentorAgent.git
cd MentorAgent
python -m venv .venv 
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
```

### 2. Launch the App
```bash
streamlit run app.py
```

### 3. Upload & Index Documents
Use the sidebar to upload PDFs, notebooks, or enter YouTube/GitHub links. Click “Build Index” to process and store chunks.

### 4. Ask Questions or Evaluate Repos
Use the tabs to:
- Chat with the mentor agent
- Evaluate a GitHub repo against bootcamp criteria
- Search indexed materials semantically

---

## 📈 Evaluation Logic

MentorAgent includes a built-in evaluator that:
- Scans public GitHub repos for key features (e.g., chunking, embeddings, retrieval)
- Scores them based on Phase One criteria
- Generates TODOs for missing components using an LLM

---

## 🧭 Future Enhancements

- 🧑‍🏫 Role-based mentoring (e.g., coding tutor, career coach)
- 🌐 Integration with Discord for real-time support.
- 🔐 Authenticated sessions and user-specific memory
- 📦 Personalized learning roadmaps


---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss your ideas.

---

## 🙌 Acknowledgments

Built with passion during the [NSK.AI Bootcamp Hackathon](https://www.linkedin.com/company/ai-nsk/). Special thanks to bootcamp organisors, and peers.

---

## Contact

For questions or support, open an issue or contact [Sumeya: lesumeya3@gmail.com](lesumeya3@gmail.com)