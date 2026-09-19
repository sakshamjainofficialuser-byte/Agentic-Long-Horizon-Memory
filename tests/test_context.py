"""Unit tests for context assembly logic in agent/context.py."""

import pytest
from agent.context import SYSTEM_INSTRUCTION, build_context
import agent.context as context_module
from memory.interface import Message


def test_system_instructions_are_included(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 1: Verify that system instructions are included at the beginning of context."""
    monkeypatch.setattr(context_module, "retrieve_relevant_memories", lambda query, session_id, limit: [])
    monkeypatch.setattr(context_module, "get_recent_messages", lambda session_id, limit: [])

    context = build_context(user_message="Hello", session_id="test_session")

    assert len(context) >= 1
    assert context[0]["role"] == "system"
    assert context[0]["content"] == SYSTEM_INSTRUCTION


def test_relevant_memories_are_included(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 2: Verify that relevant retrieved memories are included in the context."""
    mock_memories = [
        "User prefers Python 3.10+",
        "Project name is Agentic Long-Horizon Memory",
    ]
    monkeypatch.setattr(
        context_module,
        "retrieve_relevant_memories",
        lambda query, session_id, limit: mock_memories,
    )
    monkeypatch.setattr(context_module, "get_recent_messages", lambda session_id, limit: [])

    context = build_context(user_message="What is my project?", session_id="test_session")

    # Second element should be the background memories
    assert len(context) == 3
    assert context[0]["role"] == "system"
    assert context[0]["content"] == SYSTEM_INSTRUCTION

    assert context[1]["role"] == "system"
    assert "Relevant background memories:" in context[1]["content"]
    assert "User prefers Python 3.10+" in context[1]["content"]
    assert "Project name is Agentic Long-Horizon Memory" in context[1]["content"]


def test_recent_messages_are_included(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 3: Verify that recent conversation messages are included in chronological order."""
    mock_recent: list[Message] = [
        {"role": "user", "content": "Initial question"},
        {"role": "assistant", "content": "Initial answer"},
    ]
    monkeypatch.setattr(context_module, "retrieve_relevant_memories", lambda query, session_id, limit: [])
    monkeypatch.setattr(context_module, "get_recent_messages", lambda session_id, limit: mock_recent)

    context = build_context(user_message="Follow up question", session_id="test_session")

    assert len(context) == 4
    assert context[0]["role"] == "system"
    assert context[1]["role"] == "user"
    assert context[1]["content"] == "Initial question"
    assert context[2]["role"] == "assistant"
    assert context[2]["content"] == "Initial answer"
    assert context[3]["role"] == "user"
    assert context[3]["content"] == "Follow up question"


def test_current_user_message_appears_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 4: Verify that the current user message appears exactly once in the final context."""
    current_prompt = "Tell me the deadline"

    # Scenario A: get_recent_messages() only returns past messages
    past_messages: list[Message] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    monkeypatch.setattr(context_module, "retrieve_relevant_memories", lambda query, session_id, limit: [])
    monkeypatch.setattr(context_module, "get_recent_messages", lambda session_id, limit: past_messages)

    context_a = build_context(user_message=current_prompt, session_id="test_session")
    user_messages_a = [m for m in context_a if m["role"] == "user" and m["content"] == current_prompt]
    assert len(user_messages_a) == 1
    assert context_a[-1] == {"role": "user", "content": current_prompt}

    # Scenario B: get_recent_messages() includes the current user message because save_message ran first
    messages_with_duplicate: list[Message] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": current_prompt},
    ]
    monkeypatch.setattr(
        context_module,
        "get_recent_messages",
        lambda session_id, limit: messages_with_duplicate,
    )

    context_b = build_context(user_message=current_prompt, session_id="test_session")
    user_messages_b = [m for m in context_b if m["role"] == "user" and m["content"] == current_prompt]
    assert len(user_messages_b) == 1
    assert context_b[-1] == {"role": "user", "content": current_prompt}


def test_complete_context_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the strict 4-step context assembly order: System -> Memories -> Recent -> Current."""
    mock_memories = ["Fact: Project uses SQLite"]
    mock_recent: list[Message] = [
        {"role": "user", "content": "Turn 1 user"},
        {"role": "assistant", "content": "Turn 1 assistant"},
    ]
    current_user_msg = "Turn 2 user"

    monkeypatch.setattr(
        context_module,
        "retrieve_relevant_memories",
        lambda query, session_id, limit: mock_memories,
    )
    monkeypatch.setattr(context_module, "get_recent_messages", lambda session_id, limit: mock_recent)

    context = build_context(user_message=current_user_msg, session_id="session_123")

    assert len(context) == 5
    # 1. System instructions
    assert context[0] == {"role": "system", "content": SYSTEM_INSTRUCTION}
    # 2. Memories
    assert context[1]["role"] == "system"
    assert "Fact: Project uses SQLite" in context[1]["content"]
    # 3. Recent history
    assert context[2] == {"role": "user", "content": "Turn 1 user"}
    assert context[3] == {"role": "assistant", "content": "Turn 1 assistant"}
    # 4. Current user message
    assert context[4] == {"role": "user", "content": current_user_msg}
