"""FastAPI backend server for Agentic Long-Horizon Memory visual dashboard.

Provides endpoints for live chat interaction, real-time context inspection,
session management, and health monitoring.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.context import build_context
from agent.llm import LLMError
from agent.loop import run_agent

app = FastAPI(
    title="Agentic Long-Horizon Memory Dashboard",
    description="Interactive visual dashboard and API for long-horizon AI agent memory.",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory tracking of active dashboard sessions
ACTIVE_SESSIONS: dict[str, dict[str, Any]] = {}

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user prompt text")
    session_id: str = Field(default="session_default", description="Session identifier")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    context: list[dict[str, str]]
    timestamp: str


class ContextPreviewRequest(BaseModel):
    message: str = Field(..., description="The user prompt text to preview context for")
    session_id: str = Field(default="session_default", description="Session identifier")


class ContextPreviewResponse(BaseModel):
    session_id: str
    context: list[dict[str, str]]
    stage_breakdown: dict[str, Any]


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """Return backend status, configured model, and API key readiness."""
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return {
        "status": "online",
        "has_api_key": has_api_key,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/sessions")
def list_sessions() -> list[dict[str, Any]]:
    """Return all active conversation sessions and their activity metadata."""
    return list(ACTIVE_SESSIONS.values())


@app.post("/api/context/preview", response_model=ContextPreviewResponse)
def preview_context(payload: ContextPreviewRequest) -> ContextPreviewResponse:
    """Preview the 4-layer assembled context for a user prompt without executing the LLM."""
    try:
        context = build_context(
            user_message=payload.message,
            session_id=payload.session_id,
        )

        # Categorize breakdown of context layers
        system_instructions = [
            m for m in context
            if m.get("role") == "system" and "Relevant background memories:" not in m.get("content", "")
        ]
        memories = [
            m for m in context
            if m.get("role") == "system" and "Relevant background memories:" in m.get("content", "")
        ]
        recent_turns = [m for m in context[:-1] if m.get("role") != "system"]
        current_query = context[-1] if context else None

        return ContextPreviewResponse(
            session_id=payload.session_id,
            context=context,
            stage_breakdown={
                "has_system_instruction": len(system_instructions) > 0,
                "memories_count": len(memories),
                "recent_turns_count": len(recent_turns),
                "current_query": current_query,
            },
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Context assembly error: {str(err)}") from err


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Execute a turn with the agent and return the response alongside assembled context."""
    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="User message cannot be empty.")

    try:
        # Assemble context before or during turn
        context = build_context(
            user_message=user_msg,
            session_id=payload.session_id,
        )

        # Run the full agent lifecycle
        response_text = run_agent(
            user_message=user_msg,
            session_id=payload.session_id,
        )

        # Update session metadata
        sess_data = ACTIVE_SESSIONS.get(payload.session_id, {
            "session_id": payload.session_id,
            "turns": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_message": "",
        })
        sess_data["turns"] += 1
        sess_data["last_message"] = user_msg
        sess_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        ACTIVE_SESSIONS[payload.session_id] = sess_data

        return ChatResponse(
            response=response_text,
            session_id=payload.session_id,
            context=context,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except LLMError as err:
        raise HTTPException(status_code=502, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Internal agent error: {str(err)}") from err


# Mount static assets if the directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
