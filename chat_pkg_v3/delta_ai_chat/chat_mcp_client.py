from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


class MCPChatClient:
    """
    Client for the Delta AI MCP chat server (SSE/HTTP).
    Starts the server as a subprocess and calls the single MCP tool: `chat`.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, profile_name: str = "bmc-sie-prod"):
        self.host = host
        self.port = port
        self.profile_name = profile_name
        self.base_url = f"http://{host}:{port}"
        self.sse_url = f"{self.base_url}/sse"

        self._proc: Optional[subprocess.Popen] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None
        self._session_lock = asyncio.Lock()

    # def start_server_subprocess(self) -> None:
    #     if self._proc and self._proc.poll() is None:
    #         return

    #     server_module =  os.path.join(os.path.dirname(__file__), "chat_mcp_server")
    #     self._proc = subprocess.Popen(
    #         [
    #             sys.executable,
    #             "-m",
    #             server_module,
    #             "--host",
    #             self.host,
    #             "--port",
    #             str(self.port),
    #             "--profile",
    #             self.profile_name,
    #         ],
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.PIPE,
    #         text=True,
    #     )

    #     deadline = time.time() + 30
    #     last_err: Optional[Exception] = None
    #     while time.time() < deadline:
    #         try:
    #             asyncio.run(self._smoke_test())
    #             return
    #         except Exception as e:
    #             last_err = e
    #             time.sleep(0.5)

    #     raise RuntimeError(f"Failed to start MCP chat server at {self.base_url}. Last error: {last_err}")

    async def _smoke_test(self) -> None:
        async with AsyncExitStack() as stack:
            read_stream, write_stream = await stack.enter_async_context(sse_client(self.sse_url))
            session = ClientSession(read_stream, write_stream)
            await stack.enter_async_context(session)
            await session.initialize()
            tools = await session.list_tools()
            tool_names = [t.name for t in getattr(tools, "tools", [])]
            if "chat" not in tool_names:
                raise RuntimeError(f"MCP server does not expose 'chat' tool. Tools={tool_names}")

    async def _get_session(self) -> ClientSession:
        async with self._session_lock:
            if self._session:
                return self._session

            self._exit_stack = AsyncExitStack()
            read_stream, write_stream = await self._exit_stack.enter_async_context(sse_client(self.sse_url))
            self._session = ClientSession(read_stream, write_stream)
            await self._exit_stack.enter_async_context(self._session)
            await self._session.initialize()
            return self._session

    async def chat(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns dict: { "session_id": str, "response": str }
        """
        session = await self._get_session()
        args = {"session_id": session_id, "message": message}
        result = await session.call_tool(name="chat", arguments=args)

        content_items = getattr(result, "content", None) or []
        if content_items and hasattr(content_items[0], "text"):
            payload = content_items[0].text
        else:
            payload = str(result)

        try:
            data = json.loads(payload)
            # Return only what should be displayed to user
            return {"response": data.get("response", "")}
        except Exception:
            # Fallback: if server didn't return JSON, return raw string
            return {"response": payload}

    async def aclose(self) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None
