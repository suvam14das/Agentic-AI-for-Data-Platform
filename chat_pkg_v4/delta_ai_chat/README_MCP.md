# Delta AI Chat v4 – MCP tools server + LangGraph core

This package is structured so that:

1. **LangGraph is used only inside the core** (`delta_ai_chat/core.py`) to orchestrate the agent flow.
2. **All structured tools are exposed as MCP tools** via a standalone MCP server (`delta_ai_chat/tools_server.py`).
3. The **tools MCP server can be used in two ways**:
   - By the LangGraph core (agent) to execute tools (today: via local `ToolsManager`; optionally can be switched to an MCP client call pattern).
   - By external agents/clients such as **Cline** or other MCP-compatible agents, **without** going through the LangGraph agent.

> Note: `delta_ai_chat/chat_mcp_server.py` still exists as an optional server that exposes a single `chat` tool (agent).  
> The primary intended standalone MCP service is `tools_server.py` (tools only).

---

## Servers

### 1) Tools-only MCP server (recommended)

Exposes the structured tools:
- `retrieval`
- `run_sql`
- `format_to_html`
- `visualize`

Run:

```bash
python -m delta_ai_chat.tools_server --host 127.0.0.1 --port 8765 --profile bmc-sie-prod
```

Endpoints:
- `GET /sse`
- `POST /messages/`

This server is meant to be registered in any MCP client (Cline, other agents, etc).

---

### 2) Optional agent MCP server (single `chat` tool)

Exposes:
- `chat` → runs the LangGraph agent (`DeltaAIChat`)

Run:

```bash
python -m delta_ai_chat.chat_mcp_server --host 127.0.0.1 --port 8766 --profile bmc-sie-prod
```

---

## How the pieces fit

- `delta_ai_chat/tools_registry.py`
  - Single source of truth for tool specs (name/description/input schema/handler)
  - `ToolsManager.build_tool_specs()` produces the list of tools

- `delta_ai_chat/tools_server.py`
  - Publishes those tools as MCP tools (SSE/HTTP MCP server)
  - **No LangGraph usage here**

- `delta_ai_chat/core.py`
  - LangGraph orchestration for the agent
  - Uses tools from `ToolsManager` (local execution) today
  - Can be adapted to call tools via MCP (remote execution) if desired

---

## Docker / Backend notes

The existing `Dockerfile` currently starts the FastAPI backend (`chat_app_backend.app:app`).
If you want a container that exposes the MCP tools server instead, change `CMD` to:

```dockerfile
CMD ["python", "-m", "delta_ai_chat.tools_server", "--host", "0.0.0.0", "--port", "8765"]
```

Or keep both in separate containers.
