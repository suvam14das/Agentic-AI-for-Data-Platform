# Delta AI Chat MCP tools server

The standalone MCP server exposes `retrieval`, `run_sql`, and `visualize` over
Streamable HTTP. The LangGraph core can consume these tools from the same MCP
endpoint.

## Configuration

Set these environment variables for the tools server:

- `DELTA_AI_PROFILE`: OCI configuration profile (defaults to `DEFAULT`).
- `DELTA_AI_JDBC_URL`: Dataflow Spark JDBC endpoint.
- `DELTA_AI_EMBEDDING_MODEL_ID`: embedding model identifier.
- `DELTA_AI_EMBEDDING_ENDPOINT`: embedding inference endpoint.
- `DELTA_AI_EMBEDDING_COMPARTMENT_ID`: embedding compartment identifier.

The LangGraph chat core also needs `DELTA_AI_LLM_MODEL_ID`,
`DELTA_AI_LLM_ENDPOINT`, and `DELTA_AI_LLM_COMPARTMENT_ID`.

If both `DELTA_AI_REGION` and `DELTA_AI_TENANCY_NAME` are set, the application
starts or refreshes an OCI CLI session for the selected profile. Otherwise it
uses a session that you have authenticated separately. These settings are not
stored in this v4 package.

## Run

From `chat_pkg_v4`:

```bash
python -m delta_ai_chat.tools_server --host 127.0.0.1 --port 8765
```

Register `http://127.0.0.1:8765/mcp` as a Streamable HTTP MCP server in your
client. Override the bind address and port with `DELTA_AI_MCP_HOST` and
`DELTA_AI_MCP_PORT`. The LangGraph core uses `DELTA_AI_MCP_URL` to point to the
server when it is hosted elsewhere.
