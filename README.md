# Agentic Long-Horizon Memory

A modular Python framework that maintains conversational continuity, recalls important background facts, and prevents context-drift across thousands of turns.

---

## 🏗️ Repository Structure

```
agentic-long-horizon-memory/
├── agent/
│   ├── __init__.py      # Package initialization and exports
│   ├── loop.py          # Execution loop and CLI chat interface
│   ├── context.py       # 4-stage context assembly & deduplication
│   └── llm.py           # OpenAI SDK integration & error handling
├── memory/
│   ├── __init__.py      # Memory package exports
│   └── interface.py     # Clean contract for SQLite & FAISS integration
├── api/
│   ├── __init__.py      # API package initialization
│   ├── app.py           # FastAPI server & REST endpoints
│   └── static/          # Visual Web Dashboard UI
│       ├── index.html   # 3-column dashboard structure
│       ├── style.css    # Modern dark theme & glassmorphism
│       └── app.js       # Real-time context inspector client
├── tests/
│   ├── __init__.py      # Tests package initialization
│   ├── test_context.py  # Context assembly & ordering unit tests
│   ├── test_loop.py     # Execution loop & LLM error handling tests
│   └── test_api.py      # Dashboard API endpoint tests
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
├── requirements.txt     # Production and testing dependencies
└── README.md            # Documentation & integration guide
```

---

## 👥 Two-Person Architecture Division

- **Person 1 (Current Implementation)**:
  - Agent execution loop (`agent/loop.py`)
  - Context assembly and message ordering (`agent/context.py`)
  - OpenAI LLM integration with safe error handling (`agent/llm.py`)
  - Command-line chat interface (`python -m agent.loop`)
  - Visual Web Dashboard & API (`api/app.py`, `api/static/`)
  - Full unit & integration test suite (`tests/`)
- **Person 2 (Future Storage Implementation)**:
  - Persistent SQLite message storage (`save_message`, `get_recent_messages`)
  - Vector embeddings and FAISS retrieval index (`retrieve_relevant_memories`)
  - Summarization / compression background tasks

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your OpenAI API credentials:
```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-proj-your_actual_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## 🖥️ Launching the Visual Dashboard

Start the FastAPI dashboard server:

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **`http://127.0.0.1:8000`** to access:
- **Live Agent Chat**: Real-time interaction with the agent.
- **Context Inspector**: Visual inspection of the 4 assembled layers (System prompt, Retrieved memories, Recent turns, Current user prompt).
- **Session Switcher**: Manage and isolate separate conversation sessions.

---

## 🚀 Running the Terminal CLI

Launch the interactive terminal chat:

```bash
python -m agent.loop
```

- **Session ID**: Enter an existing session identifier or press `Enter` to auto-generate one.
- **Chat**: Enter messages interactively.
- **Exit**: Type `exit` or `quit` to exit cleanly.

---

## 🧪 Running Tests

Run the complete test suite with `pytest`:

```bash
pytest -v
```

All 18 unit and integration tests run deterministically with mocks and monkeypatching without making external API calls.

---

## 🔌 Memory Interface Specification (For Person 2)

Person 2 connects SQLite and FAISS by providing implementations for the functions in `memory/interface.py`:

```python
from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


def save_message(
    session_id: str,
    role: str,
    content: str,
) -> None:
    """Save a single conversation turn to SQLite."""
    ...


def get_recent_messages(
    session_id: str,
    limit: int = 10,
) -> list[Message]:
    """Retrieve chronological recent messages for the given session."""
    ...


def retrieve_relevant_memories(
    query: str,
    session_id: str,
    limit: int = 5,
) -> list[str]:
    """Retrieve semantically relevant long-term memory snippets via FAISS / Vector search."""
    ...
```

### Context Assembly Guarantees
`agent/context.py` guarantees the following payload order to the LLM:
1. **System Instructions**
2. **Relevant Long-term Memories** (if non-empty)
3. **Recent Conversation Messages** (chronological prior turns)
4. **Current User Message** (appears exactly once at the end)
