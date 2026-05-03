from __future__ import annotations

import argparse
import os

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "FastMCP is not available. Install or upgrade the 'mcp' package in the runtime environment."
    ) from exc

try:
    from delta_ai_chat.tools_registry import ToolsManager, register_tools
except ImportError:
    from tools_registry import ToolsManager, register_tools


def build_mcp_server(profile_name: str, host: str, port: int) -> FastMCP:
    """
    Build the MCP server and register all tools from tools_registry.
    """
    mcp = FastMCP(
        name="delta-ai-tools",
        instructions="Delta AI Chat tools exposed over MCP (SSE/HTTP).",
        host=host,
        port=port,
    )
    tools = ToolsManager(profile_name=profile_name)
    register_tools(mcp, tools)
    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Delta AI Chat MCP Tools Server (SSE/HTTP)")
    parser.add_argument("--host", default=os.environ.get("DELTA_AI_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DELTA_AI_MCP_PORT", "8765")))
    parser.add_argument("--profile", default=os.environ.get("DELTA_AI_PROFILE", "bmc-sie-prod"))
    args = parser.parse_args()

    mcp = build_mcp_server(profile_name=args.profile, host=args.host, port=args.port)
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()
