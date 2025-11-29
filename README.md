# LocalRAG: Self-Hosted RAG System for Code & Documents

A production-ready, Docker-powered RAG system that understands the difference between code and prose. Ingest your codebase and documentation, then query them with full privacy and zero configuration.

![Dashboard Overview](/kit/assets/dashboard.png)

---

## 🎯 Why This Exists

Most RAG systems treat all data the same—they chunk your Python files the same way they chunk your PDFs. This is a mistake.

**LocalRAG uses context-aware ingestion:**
- **Code collections** use AST-based chunking that respects function boundaries
- **Document collections** use semantic chunking optimized for prose
- **Separate collections** prevent context pollution (your API docs don't interfere with your codebase queries)

**Example:**
```bash
# Ask about your docs
"What was our Q3 strategy?" → queries the 'company_docs' collection

# Ask about your code  
"Show me the authentication middleware" → queries the 'backend_code' collection
```

This separation is what makes answers actually useful.

---

## ⚡ Quick Start (5 Minutes)

**Prerequisites:**
- Docker & Docker Compose
- [Ollama](https://ollama.com/) running locally

**Setup:**
```bash
# 1. Pull the embedding model
ollama pull nomic-embed-text

# 2. Clone and start
git clone https://github.com/2dogsandanerd/Knowledge-Base-Self-Hosting-Kit.git
cd Knowledge-Base-Self-Hosting-Kit
docker compose up -d
```

**That's it.** Open `http://localhost:8080`

---

## 🚀 Try It: Upload & Query (30 Seconds)

1. Go to the **Upload** tab
2. Upload any PDF or Markdown file
3. Go to the **Quicksearch** tab
4. Select your collection and ask a question

![Query Interface](/kit/assets/query.png)

---

## 💡 The Power Move: Analyze Your Own Codebase

Let's ingest this repository's backend code and query it like a wiki.

**Step 1: Copy code into the data folder**
```bash
# The ./data/docs folder is mounted as / in the container
cp -r backend/src data/docs/localrag_code
```

**Step 2: Ingest via UI**
- Navigate to **Folder Ingestion** tab
- Path: `/localrag_code`
- Collection: `localrag_code`
- Profile: **Codebase** (uses code-optimized chunking)
- Click **Start Ingestion**

![Folder Ingestion](/kit/assets/folder_ingest.png)

**Step 3: Query your code**
- Go to **Quicksearch**
- Select `localrag_code` collection
- Ask: *"How does the folder ingestion work?"* or *"Show me the RAGClient class"*

You'll get answers with direct code snippets. This is invaluable for:
- Onboarding new developers
- Understanding unfamiliar codebases
- Debugging complex systems

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│         Your Browser (localhost:8080)            │
└──────────────────────────┬───────────────────────┘
                           │
┌──────────────────────────▼───────────────────────┐
│              Gateway (Nginx)                     │
│  - Serves static frontend                        │
│  - Proxies /api/* to backend                     │
└──────────────────────────┬───────────────────────┘
                           │
┌──────────────────────────▼───────────────────────┐
│       Backend (FastAPI + LlamaIndex)             │
│  - REST API for ingestion & queries              │
│  - Async task management                         │
│  - Orchestrates ChromaDB & Ollama                │
└─────────────────┬──────────────────┬─────────────┘
                  │                  │
┌─────────────────▼──────┐  ┌────────▼──────────────┐
│  ChromaDB              │  │   Ollama              │
│  - Vector storage      │  │  - Embeddings         │
│  - Persistent on disk  │  │  - Answer generation  │
└────────────────────────┘  └───────────────────────┘
```

**Tech Stack:**
- **Backend:** FastAPI, LlamaIndex 0.12.9
- **Vector DB:** ChromaDB 0.5.23
- **LLM/Embeddings:** Ollama (configurable)
- **Document Parser:** Docling 2.13.0 (advanced OCR, table extraction)
- **Frontend:** Vanilla HTML/JS (no build step)

**Linux Users:** If Ollama runs on your host, you may need to set `OLLAMA_HOST=http://host.docker.internal:11434` in `.env` or use `--network host`.

---

## ✨ Features

- ✅ **100% Local & Private** — Your data never leaves your machine
- ✅ **Zero Config** — `docker compose up` and you're running
- ✅ **Batch Ingestion — Process multiple files (sequential processing in Community Edition)
- ✅ **Code & Doc Profiles** — Different chunking strategies for code vs. prose
- ✅ **Smart Ingestion** — Auto-detects file types, avoids duplicates
- ✅ **`.ragignore` Support** — Works like `.gitignore` to exclude files/folders
- ✅ **Full REST API** — Programmatic access for automation

---

## 🐍 API Example

```python
import requests
import time

BASE_URL = "http://localhost:8080/api/v1/rag"

# 1. Create a collection
print("Creating collection...")
requests.post(f"{BASE_URL}/collections", json={"collection_name": "api_docs"})

# 2. Upload a document
print("Uploading README.md...")
with open("README.md", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/documents/upload",
        files={"files": ("README.md", f, "text/markdown")},
        data={"collection_name": "api_docs"},
    ).json()

task_id = response.get("task_id")
print(f"Task ID: {task_id}")

# 3. Poll for completion
while True:
    status = requests.get(f"{BASE_URL}/ingestion/ingest-status/{task_id}").json()
    print(f"Status: {status['status']}, Progress: {status['progress']}%")
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(2)

# 4. Query
print("\nQuerying...")
result = requests.post(
    f"{BASE_URL}/query",
    json={"query": "What is the killer feature?", "collection": "api_docs", "k": 3},
).json()

print("\nAnswer:")
print(result.get("answer"))

print("\nSources:")
for source in result.get("metadata", []):
    print(f"- {source.get('filename')}")
```

---

## 🔧 Configuration

Create a `.env` file to customize:

```env
# Change the public port
PORT=8090

# Swap LLM/embedding models
LLM_PROVIDER=ollama
LLM_MODEL=llama3:8b
EMBEDDING_MODEL=nomic-embed-text

# Use OpenAI/Anthropic instead
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
```

See `.env.example` for all options.

---

## 👨‍💻 Development

**Hot-Reloading:**  
The backend uses Uvicorn's auto-reload. Edit files in `backend/src` and changes apply instantly.

**Rebuild after dependency changes:**
```bash
docker compose up -d --build backend
```

**Project Structure:**
```
localrag/
├── backend/
│   ├── src/
│   │   ├── api/          # FastAPI routes
│   │   ├── core/         # RAG logic (RAGClient, services)
│   │   ├── models/       # Pydantic models
│   │   └── main.py       # Entry point
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/             # Static HTML/JS
├── nginx/                # Reverse proxy config
├── data/                 # Mounted volume for ingestion
└── docker-compose.yml
```

---

## 🧪 Advanced: Multi-Collection Search

You can query across multiple collections simultaneously:

```python
result = requests.post(
    f"{BASE_URL}/query",
    json={
        "query": "How do we handle authentication?",
        "collections": ["backend_code", "api_docs"],  # Note: plural
        "k": 5
    }
).json()
```

This is useful when answers might span code and documentation.

---

## 📊 What Makes This Different?

| Feature | LocalRAG | Typical RAG |
|---------|----------|-------------|
| **Code-aware chunking** | ✅ AST-based | ❌ Fixed-size |
| **Context separation** | ✅ Per-collection profiles | ❌ One-size-fits-all |
| **Self-hosted** | ✅ 100% local | ⚠️ Often cloud-dependent |
| **Zero config** | ✅ Docker Compose | ❌ Complex setup |
| **Async ingestion** | ✅ Background tasks | ⚠️ Varies |
| **Production-ready** | ✅ FastAPI + ChromaDB | ⚠️ Often prototypes |

---

## 🚧 Roadmap

- [ ] Support for more LLM providers (Anthropic, Cohere)
- [ ] Advanced reranking (Cohere Rerank, Cross-Encoder)
- [ ] Multi-modal support (images, diagrams)
- [ ] Graph-based retrieval for code dependencies
- [ ] Evaluation metrics dashboard (RAGAS integration)

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Built With

- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [LlamaIndex](https://www.llamaindex.ai/) — RAG orchestration
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [Ollama](https://ollama.com/) — Local LLM runtime
- [Docling](https://github.com/DS4SD/docling) — Advanced document parsing

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 💬 Questions?

- **Issues:** [GitHub Issues](https://github.com/2dogsandanerd/Knowledge-Base-Self-Hosting-Kit/issues)
- **Discussions:** [GitHub Discussions](https://github.com/2dogsandanerd/Knowledge-Base-Self-Hosting-Kit/discussions)

---

**⭐ If you find this useful, please star the repo!**
