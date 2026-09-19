"""Unit tests for the agent execution loop and LLM error handling."""

from unittest.mock import MagicMock
import openai
import pytest
import agent.llm as llm_module
import agent.loop as loop_module
from agent.llm import LLMError, call_llm
from agent.loop import run_agent


def test_run_agent_saves_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 5: Verify that run_agent() saves the incoming user message."""
    saved_calls: list[dict[str, str]] = []

    def mock_save_message(session_id: str, role: str, content: str) -> None:
        saved_calls.append({"session_id": session_id, "role": role, "content": content})

    monkeypatch.setattr(loop_module, "save_message", mock_save_message)
    monkeypatch.setattr(
        loop_module,
        "build_context",
        lambda user_message, session_id, **kwargs: [{"role": "user", "content": user_message}],
    )
    monkeypatch.setattr(loop_module, "call_llm", lambda ctx: "Assistant reply")

    run_agent(user_message="My test message", session_id="session_abc")

    # Verify first save call was the user message
    assert len(saved_calls) == 2
    assert saved_calls[0] == {
        "session_id": "session_abc",
        "role": "user",
        "content": "My test message",
    }


def test_run_agent_calls_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 6: Verify that run_agent() invokes call_llm with the assembled context."""
    mock_context = [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "Question"},
    ]
    llm_received_context: list[dict[str, str]] = []

    def mock_call_llm(context: list[dict[str, str]]) -> str:
        llm_received_context.extend(context)
        return "Generated answer"

    monkeypatch.setattr(loop_module, "save_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        loop_module,
        "build_context",
        lambda user_message, session_id, **kwargs: mock_context,
    )
    monkeypatch.setattr(loop_module, "call_llm", mock_call_llm)

    response = run_agent(user_message="Question", session_id="session_abc")

    assert response == "Generated answer"
    assert llm_received_context == mock_context


def test_run_agent_saves_assistant_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 7: Verify that run_agent() saves the assistant response to memory."""
    saved_calls: list[dict[str, str]] = []

    def mock_save_message(session_id: str, role: str, content: str) -> None:
        saved_calls.append({"session_id": session_id, "role": role, "content": content})

    monkeypatch.setattr(loop_module, "save_message", mock_save_message)
    monkeypatch.setattr(
        loop_module,
        "build_context",
        lambda user_message, session_id, **kwargs: [],
    )
    monkeypatch.setattr(loop_module, "call_llm", lambda ctx: "Generated response text")

    result = run_agent(user_message="Hello", session_id="session_xyz")

    assert result == "Generated response text"
    assert len(saved_calls) == 2
    assert saved_calls[1] == {
        "session_id": "session_xyz",
        "role": "assistant",
        "content": "Generated response text",
    }


def test_api_connection_error_converted_to_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 8a: Verify APIConnectionError is caught and raised as LLMError."""
    monkeypatch.setenv("OPENAI_API_KEY", "mock-key")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
        request=MagicMock()
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda: mock_client)

    with pytest.raises(LLMError) as exc_info:
        call_llm([{"role": "user", "content": "Hi"}])

    assert "Could not connect to OpenAI API" in str(exc_info.value)


def test_api_rate_limit_error_converted_to_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 8b: Verify RateLimitError is caught and raised as LLMError."""
    monkeypatch.setenv("OPENAI_API_KEY", "mock-key")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        message="Rate limit reached",
        response=MagicMock(status_code=429),
        body=None,
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda: mock_client)

    with pytest.raises(LLMError) as exc_info:
        call_llm([{"role": "user", "content": "Hi"}])

    assert "rate limit or quota exceeded" in str(exc_info.value)


def test_api_status_error_converted_to_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 8c: Verify APIStatusError is caught and raised as LLMError."""
    monkeypatch.setenv("OPENAI_API_KEY", "mock-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_client.chat.completions.create.side_effect = openai.APIStatusError(
        message="Internal server error",
        response=mock_response,
        body=None,
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda: mock_client)

    with pytest.raises(LLMError) as exc_info:
        call_llm([{"role": "user", "content": "Hi"}])

    assert "HTTP 500" in str(exc_info.value)


def test_missing_api_key_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing OPENAI_API_KEY raises a clean LLMError without exposing internal trace."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMError) as exc_info:
        call_llm([{"role": "user", "content": "Hi"}])

    assert "OPENAI_API_KEY is not set" in str(exc_info.value)
