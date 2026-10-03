# Agentic AI for Data Platform

Agentic AI workflows for a data platform. Each `chat_pkg_v*` directory contains
one version of the chat application.

## Local configuration

Set deployment values in your environment before running a package. Do not
commit profiles, credentials, service URLs, model identifiers, JDBC URLs, saved
chat history, or generated vector stores.

- `DELTA_AI_PROFILE` (defaults to `DEFAULT`)
- `DELTA_AI_REGION` and `DELTA_AI_TENANCY_NAME` (optional, for OCI CLI session authentication)
- `DELTA_AI_JDBC_URL`
- `DELTA_AI_LLM_MODEL_ID`, `DELTA_AI_LLM_ENDPOINT`, `DELTA_AI_LLM_COMPARTMENT_ID`
- `DELTA_AI_EMBEDDING_MODEL_ID`, `DELTA_AI_EMBEDDING_ENDPOINT`, `DELTA_AI_EMBEDDING_COMPARTMENT_ID`

For the v4 MCP server, see [MCP setup](chat_pkg_v4/delta_ai_chat/README_MCP.md).
