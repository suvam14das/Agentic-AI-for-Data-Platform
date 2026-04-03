import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os
import sys
from delta_ai_chat.core import DeltaAIChat
from fastapi.staticfiles import StaticFiles

app = FastAPI()

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
        "chat": DeltaAIChat(),
        "history": []
    }
    return {"session_id": session_id}

@app.post("/chat/{session_id}")
def chat(session_id: str, request: ChatRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    user_message = request.message
    
    # Process special commands
    if user_message.lower() == "memorize":
        session["chat"].save_summary()
        response = "History saved."
    elif user_message.lower() == "exit":
        session["chat"].close()
        del sessions[session_id]
        response = "Session closed."
    elif "run sql" in user_message.lower() or "get data" in user_message.lower():
        if session["chat"].current_sql:
            response_text, new_sql = session["chat"].execute_query(session["chat"].current_sql, user_message)
            if new_sql:
                session["chat"].current_sql = new_sql
            response = response_text
        else:
            response = "No SQL query available."
    else:
        response_text, new_sql = session["chat"].get_agent_response(user_message)
        if new_sql:
            session["chat"].current_sql = new_sql
        response = response_text
    
    # Update history
    session["history"].append({"user": user_message, "ai": response})
    
    return {"response": response}

@app.get("/history/{session_id}")
def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"history": sessions[session_id]["history"]}

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chat_app_frontend')
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
