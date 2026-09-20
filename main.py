"""Main web entrypoint for Vercel deployment and local execution."""

from datetime import datetime, timezone
import os
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agent.context import build_context
from agent.llm import LLMError
from agent.loop import run_agent

app = FastAPI(
    title="Agentic Long-Horizon Memory API",
    description="Serverless API and interactive interface for Agentic Long-Horizon Memory.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user prompt text")
    session_id: str = Field(default="session_default", description="Session identifier")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    context: list[dict[str, str]]
    timestamp: str


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return {
        "status": "online",
        "has_api_key": has_api_key,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Execute a turn with the agent and return the response."""
    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        context = build_context(user_message=user_msg, session_id=payload.session_id)
        response_text = run_agent(user_message=user_msg, session_id=payload.session_id)

        return ChatResponse(
            response=response_text,
            session_id=payload.session_id,
            context=context,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except LLMError as err:
        raise HTTPException(status_code=502, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(err)}") from err


@app.get("/", response_class=HTMLResponse)
def home_page() -> str:
    """Serve a clean interactive web interface."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agentic Long-Horizon Memory</title>
  <style>
    :root {
      --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --muted: #94a3b8;
      --primary: #6366f1; --primary-hover: #4f46e5;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg); color: var(--text); margin: 0; padding: 2rem;
      display: flex; justify-content: center; align-items: center; min-height: 90vh;
    }
    .container {
      width: 100%; max-width: 720px; background: var(--card); border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.1); padding: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    h1 { margin-top: 0; font-size: 1.5rem; display: flex; align-items: center; gap: 0.5rem; }
    p.subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }
    .chat-box {
      height: 340px; overflow-y: auto; background: var(--bg); border-radius: 10px;
      padding: 1rem; border: 1px solid rgba(255,255,255,0.05); display: flex; flex-direction: column; gap: 0.75rem;
    }
    .msg { padding: 0.6rem 0.9rem; border-radius: 8px; max-width: 80%; font-size: 0.9rem; line-height: 1.4; }
    .msg.user { align-self: flex-end; background: var(--primary); color: #fff; }
    .msg.agent { align-self: flex-start; background: #334155; color: var(--text); }
    .input-row { display: flex; gap: 0.5rem; margin-top: 1rem; }
    input[type="text"] {
      flex: 1; background: var(--bg); border: 1px solid rgba(255,255,255,0.15);
      border-radius: 8px; padding: 0.75rem 1rem; color: #fff; outline: none;
    }
    button {
      background: var(--primary); color: #fff; border: none; border-radius: 8px;
      padding: 0.75rem 1.5rem; font-weight: 600; cursor: pointer;
    }
    button:hover { background: var(--primary-hover); }
    .badge {
      display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
      background: rgba(16,185,129,0.2); color: #34d399; font-size: 0.75rem; font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🧠 Agentic Long-Horizon Memory <span class="badge">Online</span></h1>
    <p class="subtitle">AI agent with 4-layer persistent memory & FAISS vector search.</p>
    <div class="chat-box" id="chat-box">
      <div class="msg agent">Hello! How can I assist you with your project today?</div>
    </div>
    <div class="input-row">
      <input type="text" id="msg-input" placeholder="Type your message..." />
      <button id="send-btn">Send</button>
    </div>
  </div>
  <script>
    const chatBox = document.getElementById('chat-box');
    const msgInput = document.getElementById('msg-input');
    const sendBtn = document.getElementById('send-btn');
    const sessionId = "session_" + Math.random().toString(36).substring(2, 9);

    async function send() {
      const text = msgInput.value.trim();
      if (!text) return;
      msgInput.value = '';

      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      userDiv.textContent = text;
      chatBox.appendChild(userDiv);
      chatBox.scrollTop = chatBox.scrollHeight;

      const loadingDiv = document.createElement('div');
      loadingDiv.className = 'msg agent';
      loadingDiv.textContent = 'Thinking...';
      chatBox.appendChild(loadingDiv);

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ message: text, session_id: sessionId })
        });
        const data = await res.json();
        loadingDiv.textContent = data.response || data.detail || 'No response';
      } catch(err) {
        loadingDiv.textContent = 'Error: ' + err.message;
      }
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    sendBtn.addEventListener('click', send);
    msgInput.addEventListener('keydown', (e) => { if(e.key === 'Enter') send(); });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
