from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount, Route

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport

try :
    from delta_ai_chat.tools_registry import ToolsManager
except ImportError:
    from tools_registry import ToolsManager



def _tool_to_mcp_types_tool(spec) -> types.Tool:
    # Pydantic v2 schema: model_json_schema()
    input_schema: Dict[str, Any] = spec.input_model.model_json_schema()
    return types.Tool(
        name=spec.name,
        description=spec.description,
        inputSchema=input_schema,
    )


def _build_app(profile_name: str) -> Starlette:
    """
    MCP SSE/HTTP server exposing the tool registry using low-level `mcp` server APIs.
    """
    tools = ToolsManager(profile_name=profile_name)
    tool_specs = tools.build_tool_specs()

    mcp_server = Server(name="delta-ai-tools", instructions="Delta AI Chat tools exposed over MCP (SSE/HTTP).")
    sse = SseServerTransport("/messages/")

    @mcp_server.list_tools()
    async def _list_tools() -> List[types.Tool]:
        return [_tool_to_mcp_types_tool(spec) for spec in tool_specs]

    @mcp_server.call_tool()
    async def _call_tool(name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        
        spec = next((s for s in tool_specs if s.name == name), None)
        if spec is None:
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
            )

        try:
            model = spec.input_model.model_validate(arguments)
            result = spec.handler(model)

            # Result is string in our tools; return as text content
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(result))],
            )
        except Exception as e:
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=f"Tool execution error: {e}")],
            )

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await mcp_server.run(
                streams[0],
                streams[1],
                mcp_server.create_initialization_options(),
            )

    routes = [
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
    ]

    return Starlette(routes=routes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delta AI Chat MCP Tools Server (SSE/HTTP)")
    parser.add_argument("--host", default=os.environ.get("DELTA_AI_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DELTA_AI_MCP_PORT", "8765")))
    parser.add_argument("--profile", default=os.environ.get("DELTA_AI_PROFILE", "bmc-sie-prod"))
    args = parser.parse_args()

    app = _build_app(profile_name=args.profile)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
