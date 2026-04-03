import sys
import os
import uvicorn
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from delta_ai_chat.chat_mcp_client import MCPChatClient

mcp_client: MCPChatClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_client
    mcp_client = MCPChatClient()
    # Starts server subprocess once for the whole app.
    # mcp_client.start_server_subprocess()
    try:
        yield
    finally:
        if mcp_client:
            await mcp_client.aclose()
            mcp_client = None


app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local session storage
sessions = {}

class ChatRequest(BaseModel):
    message: str

class SessionResponse(BaseModel):
    session_id: str

@app.post("/new_session", response_model=SessionResponse)
def new_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "history": []
    }
    return {"session_id": session_id}

@app.post("/chat/{session_id}")
async def chat(session_id: str, request: ChatRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    user_message = request.message

    global mcp_client
    if mcp_client is None:
        raise HTTPException(status_code=500, detail="MCP chat client not initialized")

    # Process special commands
    if user_message.lower() == "exit":
        del sessions[session_id]
        response = "Session closed."
    else:
        payload = await mcp_client.chat(message=user_message, session_id=session_id)
        response = payload.get("response", "")
    
    # Update history
    session["history"].append({"user": user_message, "ai": response})
    
    return {"response": response}

@app.get("/history/{session_id}")
def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"history": sessions[session_id]["history"]}

# charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "delta_ai_chat", "tmp")
# app.mount("/charts", StaticFiles(directory=charts_dir), name="charts")

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chat_app_frontend')
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delta AI Chat MCP Server (SSE/HTTP, single chat tool)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default= int("8000"))
    
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
