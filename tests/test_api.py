import pytest

try:
    from fastapi.testclient import TestClient
    from api.app import app
    import api.app as api_module
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    TestClient = None  # type: ignore
    app = None  # type: ignore
    api_module = None  # type: ignore

from agent.llm import LLMError

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI or test dependencies not installed")


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a test client for the FastAPI app."""
    return TestClient(app)


def test_health_check_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /api/health returns online status and configured model."""
    monkeypatch.setenv("OPENAI_API_KEY", "mock-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["has_api_key"] is True
    assert data["model"] == "gpt-4o-mini"


def test_context_preview_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /api/context/preview correctly breaks down context stages."""
    response = client.post(
        "/api/context/preview",
        json={"message": "What is the project deadline?", "session_id": "test_sess"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_sess"
    assert len(data["context"]) >= 2
    assert data["context"][0]["role"] == "system"
    assert data["context"][-1]["role"] == "user"
    assert data["context"][-1]["content"] == "What is the project deadline?"


def test_chat_endpoint_successful_turn(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /api/chat invokes run_agent and returns response with context."""
    monkeypatch.setattr(
        api_module,
        "run_agent",
        lambda user_message, session_id: "I have stored that deadline in memory.",
    )

    response = client.post(
        "/api/chat",
        json={"message": "The deadline is Friday.", "session_id": "sess_unit_test"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "I have stored that deadline in memory."
    assert data["session_id"] == "sess_unit_test"
    assert len(data["context"]) >= 1


def test_chat_endpoint_empty_message_rejected(client: TestClient) -> None:
    """Verify /api/chat rejects empty or whitespace-only prompts."""
    response = client.post(
        "/api/chat",
        json={"message": "   ", "session_id": "test_sess"},
    )
    assert response.status_code == 400
    assert "User message cannot be empty" in response.json()["detail"]


def test_chat_endpoint_llm_error_handling(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /api/chat returns HTTP 502 with safe message when LLMError occurs."""
    def mock_failing_run(user_message: str, session_id: str) -> str:
        raise LLMError("Rate limit exceeded")

    monkeypatch.setattr(api_module, "run_agent", mock_failing_run)

    response = client.post(
        "/api/chat",
        json={"message": "Hello", "session_id": "test_sess"},
    )
    assert response.status_code == 502
    assert "Rate limit exceeded" in response.json()["detail"]


def test_list_sessions_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /api/sessions returns active sessions after chat."""
    monkeypatch.setattr(
        api_module,
        "run_agent",
        lambda user_message, session_id: "Response",
    )

    client.post("/api/chat", json={"message": "Test 1", "session_id": "sess_listing"})
    response = client.get("/api/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert any(s["session_id"] == "sess_listing" for s in sessions)
