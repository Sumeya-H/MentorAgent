NSK.AI Mentor Agent An AI-powered teaching assistant built to support bootcamp participants and mentors at scale. It helps students navigate study materials, get instant project feedback, and reuse knowledge from past cohorts — all in one place.

🚩 The Problem Bootcamp cohorts face recurring challenges:

New participants feel lost navigating tons of study materials, recordings, and GitHub repos. Mentors are overwhelmed and can’t provide personalized support to every student. Project submissions often miss criteria due to unclear requirements. Valuable knowledge from past cohorts gets lost. 👉 Result: Students struggle, mentors burn out, and institutional knowledge disappears.

✅ Our Solution The NSK.AI Mentor Agent supports bootcamp participants 24/7 through:

Q&A Agent Answers technical & non-technical questions from:

Bootcamp study materials Session recordings (YouTube transcripts) Public GitHub repos Project requirements & rubrics Project Evaluator Agent Students submit their GitHub repo → Agent checks criteria → Provides feedback:

✅ Criteria met ❌ Missing components 💡 Suggestions for improvement Knowledge Search Semantic search across past projects & sessions:

“Show me a Phase One project with RAG + Pinecone.” “Find all sessions about Agentic AI.” 🧩 How It Works Demo Flow Student logs in → Asks “What’s the difference between Phase One and Hackathon requirements?”

Agent retrieves docs → Returns clear, cited answer. Student uploads repo → Agent evaluates against checklist.

Example: Checks for RAG, embeddings, Streamlit imports. Returns ✅/❌ + improvement tips. Student searches past work → Finds relevant repos & transcripts.

🛠 Tech Stack LLM Frameworks: LangChain, LlamaIndex Vector DB: FAISS (for hackathon), Pinecone/Weaviate (scalable) Frontend: Streamlit Repo Analysis: GitHub API, Python (AST, regex, tree-sitter) Agents: Multi-agent orchestration (Q&A, Repo Evaluator, Coordinator) 📂 Project Structure NSK.AI-Mentor-Agent/ │── data/ # Study materials, transcripts, rubrics │── notebooks/ # Prototyping and experiments │── src/ │ ├── retriever/ # Indexing + retrieval pipeline │ ├── agents/ # Q&A, Repo Evaluator, Coordinator │ ├── ui/ # Streamlit app │ └── utils/ # Helper functions │── requirements.txt │── README.md

🚀 Setup & Installation Clone the repo:

git clone https://github.com//NSK.AI-Mentor-Agent.git cd NSK.AI-Mentor-Agent Create and activate environment: python -m venv venv source venv/bin/activate # Mac/Linux venv\Scripts\activate # Windows

Install dependencies pip install -r requirements.txt

Run the app streamlit run src/ui/app.py

📌 Features Roadmap

Q&A Agent with bootcamp resources

Repo Evaluator Agent (criteria checks + LLM feedback)

Knowledge Search (projects + sessions)

Personalized learning roadmaps

Discord bot integration for real-time support

Scalable vector DB (Pinecone/Weaviate)

🌍 Impact

Students get personalized, on-demand support.

Mentors save hours of repetitive explanations.

Future cohorts benefit from a growing knowledge hub.

🤝 Contributors

Sumeya Hussein – Data prep, retrieval pipeline

dr9amxplorer – Repo Evaluator Agent

Rose Onyango – Streamlit UI, testing & polish

📄 License

This project is for educational and hackathon purposes under the NSK.AI initiative. Future licensing to be determined.

