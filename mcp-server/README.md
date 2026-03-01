# Knowledge Base Self-Hosting Kit MCP Server

A small MCP server (`knowledge-kit-mcp`) that lets OpenClaw, Smithery, or other MCP-aware agents query the same knowledge base you run locally.

## Quick start

1. Start the Knowledge Base Self-Hosting Kit backend + frontend (`docker compose up -d`).
2. Run the MCP server close to the agent:
   ```bash
   openclaw mcp add --transport stdio knowledge-kit npx -y @knowledge-kit/mcp-server
   ```
   This installs `knowledge-kit-mcp` from npm and wires MCP to your agent.
3. The agent now exposes two tools: `query_knowledge` and `list_collections`.

## Configuration

| Env var | Purpose | Default |
|---------|---------|---------|
| `KNOWLEDGE_BASE_API_URL` | HTTP URL of the running kit backend | `http://localhost:8080` |
| `KNOWLEDGE_BASE_TIMEOUT` | Request timeout in milliseconds (used for slow local LLMs) | `120000` |
| `LOG_LEVEL` | Logging level for debugging | `INFO` |

All values can also be edited from the frontend’s **Agent Configuration** tab (the UI calls `/api/v1/config`).

## Running locally

```bash
cd mcp-server
npm install
npm run build
npm start                # CLI: knowledge-kit-mcp -> build/cli.js
npm run dev              # Development mode (ts-node)
npm run start:http        # HTTP + SSE transport for web-based agents
```

## MCP tooling

- `query_knowledge(query, collections?, k?, use_reranker?)` returns LLM answers plus citations.
- `list_collections()` shows the collection names that are already ingested.

The server card (`.well-known/mcp/server-card.json`) is already configured for shared schema discovery.
