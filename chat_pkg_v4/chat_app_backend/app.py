import json
import sys
import os
import uvicorn
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from delta_ai_chat.core import DeltaAIChat

chat_core: DeltaAIChat | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_core
    chat_core = DeltaAIChat()
    try:
        yield
    finally:
        if chat_core:
            await chat_core.aclose()
            chat_core = None


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

    global chat_core
    if chat_core is None:
        raise HTTPException(status_code=500, detail="Chat core not initialized")

    # Process special commands
    if user_message.lower() == "exit":
        del sessions[session_id]
        response = "Session closed."
    else:
        response, _ = await chat_core.get_agent_response(user_message)

        # If response contains an artifact payload, inject a download_url based on current host/port.
        # This avoids hardcoding URL/port into the tool/mcp server.
        #
        # Note: in non-__main__ execution, args may not exist, so we use request.base_url.
        try:
            import re
            m = re.search(r"<artifact>(.*?)</artifact>", response, flags=re.DOTALL)
            if m:
                artifact = json.loads(m.group(1))
                if (
                    isinstance(artifact, dict)
                    and artifact.get("status") == "ok"
                    and artifact.get("type") == "table"
                    and artifact.get("format") == "csv"
                    and artifact.get("csv_filename")
                ):
                    base_url = str(request.base_url).rstrip("/")
                    artifact["download_url"] = f"{base_url}/artifacts/{artifact['csv_filename']}"
                    response = re.sub(
                        r"<artifact>.*?</artifact>",
                        f"<artifact>{json.dumps(artifact)}</artifact>",
                        response,
                        flags=re.DOTALL,
                    )
        except Exception:
            pass
    
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

# Serve CSV artifacts (local filesystem) for browser download/rendering
artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=artifacts_dir), name="artifacts")


@app.get("/artifacts_download/{filename}")
def artifacts_download(filename: str):
    # Basic path traversal protection
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = os.path.join(artifacts_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(file_path, media_type="text/csv", filename=filename)

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_app_frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delta AI Chat MCP Server (SSE/HTTP, single chat tool)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default= int("8000"))
    
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
