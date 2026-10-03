from __future__ import annotations

import argparse
import os

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as exc:
    raise ImportError(
        "MCPServer is not available. Install 'mcp>=2,<3' in the runtime environment."
    ) from exc

try:
    from delta_ai_chat.tools_registry import ToolsManager, register_tools
except ImportError:
    from tools_registry import ToolsManager, register_tools


def build_mcp_server(profile_name: str) -> MCPServer:
    """Build the MCP server and register all tools from tools_registry."""
    mcp = MCPServer(
        name="delta-ai-tools",
        instructions="Delta AI Chat tools exposed over MCP (Streamable HTTP).",
    )
    tools = ToolsManager(profile_name=profile_name)
    register_tools(mcp, tools)
    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Delta AI Chat MCP Tools Server (Streamable HTTP)")
    parser.add_argument("--host", default=os.environ.get("DELTA_AI_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DELTA_AI_MCP_PORT", "8765")))
    parser.add_argument("--profile", default=os.environ.get("DELTA_AI_PROFILE", "DEFAULT"))
    args = parser.parse_args()

    mcp = build_mcp_server(profile_name=args.profile)
    mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
