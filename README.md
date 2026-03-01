# Knowledge Base Self-Hosting Kit

[![Version](https://img.shields.io/badge/version-1.2.0-blue)](https://github.com/2dogsandanerd/Knowledge-Base-Self-Hosting-Kit/releases) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Docker Ready](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)

**A production-ready, self-hosted RAG memory layer that pairs with local LLMs, chains, and autonomous agents.**

- **ChromaDB + Docling** powers ingestion of everything from PDFs to code repositories.
- **FastAPI backend** exposes CRUD/ingestion/search APIs plus a log-friendly health endpoint.
- **Modern UI** includes dashboards, ingestion tooling, and now a dedicated **Agent Configuration** tab for MCP clients.

---

## 🚀 Quick Start

1. `git clone https://github.com/2dogsandanerd/Knowledge-Base-Self-Hosting-Kit.git`
2. `cd Knowledge-Base-Self-Hosting-Kit`
3. `cp .env.example .env` and adjust `DOCS_DIR`, LLM provider settings, etc.
4. `docker compose up -d`
5. `curl http://localhost:8080/health` to check the backend.
6. Open `http://localhost:8080` in your browser.

**Services (all exposed via the root nginx gateway):**
- Web UI + Agent Configuration tab: `http://localhost:8080/`
- OpenAPI docs: `http://localhost:8080/docs`
- Health check: `http://localhost:8080/health`
- API root: `http://localhost:8080/api/v1/rag`

---

## ⚙️ Features at a Glance

- **Ingestion modes:** single uploads, folder scans, hybrid chunking (vector + BM25).
- **Agent-ready search:** run semantic queries with `k` tuning, view citations, and see streaming logs.
- **MCP integration:** connect OpenClaw or other MCP-aware agents via the included MCP server.
- **Agent Configuration tab:** configure MCP connection details without touching `.env` or the CLI.
- **Local LLM friendly:** default Ollama setup shares host models, but OpenAI-compatible servers are supported via `OPENAI_BASE_URL`.

---

## 🧠 Agent Configuration (New Tab)

The UI now surfaces an **Agent Configuration** tab where you can:
- Review the MCP-friendly CLI snippet.
- Set `KNOWLEDGE_BASE_API_URL`, timeouts, and log levels for the MCP server directly.
- Test connectivity before handing the config to your agent.

### Recommended OpenClaw command

```
openclaw mcp add --transport stdio knowledge-kit npx -y @knowledge-kit/mcp-server
```

Existing agents can point at the tab’s configuration values (copyable) or use the CLI above to bootstrap the connector.

---

## 📁 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | External port for the nginx gateway |
| `DOCS_DIR` | `./data/docs` | Host path mounted at `/host_root` for folder ingestion |
| `LLM_PROVIDER` | `ollama` | Model provider (ollama, openai, anthropic, gemini, openai_compatible) |
| `LLM_MODEL` | `llama3:latest` | Selected model for LLM calls |
| `EMBEDDING_PROVIDER` | `ollama` | Embedding provider (usually same as LLM) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `CHUNK_SIZE` | `512` | Document chunk size |
| `CHUNK_OVERLAP` | `128` | Chunk overlap for context continuity |
| `DEBUG` | `false` | Toggle verbose logging |
| `LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARN`/`ERROR` |

For OpenAI-compatible servers, set `LLM_PROVIDER=openai_compatible` and point `OPENAI_BASE_URL` to your local endpoint.

---

## 🔍 Ingestion & Search API

- `POST /api/v1/rag/collections`: create a collection with embedding metadata.
- `POST /api/v1/rag/documents/upload`: upload PDF/MD/TXT files.
- `POST /api/v1/rag/ingest-folder`: kick off folder ingestion (uploads via mounted `DOCS_DIR`).
- `GET /api/v1/rag/collections/:name/stats`: inspect document counts.
- `POST /api/v1/rag/query`: semantic queries (vector + BM25 fusion) with citation data.
- `POST /api/v1/rag/search`: immediate keyword search without LLM generation.

Use the frontend or any HTTP client; JSON responses include `answer`, `sources`, and scoring metadata.

---

## 🧪 MCP Configuration API

- `GET /api/v1/config`: read current `.env` values via `ConfigService`.
- `POST /api/v1/config`: persist updates sent from the UI.
- `POST /api/v1/config/test`: validate downstream connections (LLM, Ollama, etc.) before deploying the change.

This powers the new **Agent Configuration** tab so you can tune the connector without restarting containers.

---

## 🛠️ Development

- Backend: `cd backend && pip install -r requirements.txt && uvicorn src.main:app --reload`.
- Frontend is static; editing `frontend/index.html` suffices.
- Build stack with Docker: `docker compose build` to refresh containers.
- Configuration persists in `.env`, but the UI uses `/api/v1/config` for live editing.

### MCP Server

The companion connector lives in `mcp-server/` and publishes `@knowledge-kit/mcp-server`. Run it with:
```
cd mcp-server
npm install
npm run build
npm start
```

---

## 📚 Documentation & Design Notes

- Architecture diagrams, MCP guides, and customization tips live under `docs/`.
- `CHROMADB` and `Ollama` containers are orchestrated via `docker-compose.yml`.
- Use `smithery/` configs to run the MCP server inside Smithery.

---

## ❤️ Contributing

1. Open an issue describing your feature or bug.
2. Create a branch off `main` and target your change at the repo’s layout.
3. Keep docs, tests, and UI changes in sync when you refactor functionality.
4. If you update MCP wiring, ensure the Agent Configuration tab and `mcp-server` read the same env names.
