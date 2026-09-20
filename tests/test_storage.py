"""Unit tests for SQLite persistent storage layer."""

import gc
import os
import tempfile
import pytest
from memory.storage import get_recent_messages_sqlite, init_db, save_message_sqlite


@pytest.fixture
def temp_db_path() -> str:
    """Create a temporary database file for test isolation."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass


def test_sqlite_persistence_and_ordering(temp_db_path: str) -> None:
    """Verify messages are saved to SQLite and retrieved in chronological order."""
    init_db(temp_db_path)

    save_message_sqlite("sess_1", "user", "Message 1", db_path=temp_db_path)
    save_message_sqlite("sess_1", "assistant", "Response 1", db_path=temp_db_path)
    save_message_sqlite("sess_1", "user", "Message 2", db_path=temp_db_path)

    messages = get_recent_messages_sqlite("sess_1", limit=10, db_path=temp_db_path)

    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "Message 1"}
    assert messages[1] == {"role": "assistant", "content": "Response 1"}
    assert messages[2] == {"role": "user", "content": "Message 2"}


def test_sqlite_limit_enforcement(temp_db_path: str) -> None:
    """Verify limit returns only the most recent N messages in chronological order."""
    init_db(temp_db_path)

    for i in range(10):
        save_message_sqlite("sess_limit", "user", f"Turn {i}", db_path=temp_db_path)

    recent = get_recent_messages_sqlite("sess_limit", limit=3, db_path=temp_db_path)

    assert len(recent) == 3
    assert recent[0]["content"] == "Turn 7"
    assert recent[1]["content"] == "Turn 8"
    assert recent[2]["content"] == "Turn 9"


def test_sqlite_session_isolation(temp_db_path: str) -> None:
    """Verify messages from different sessions are completely isolated."""
    init_db(temp_db_path)

    save_message_sqlite("session_A", "user", "Secret for A", db_path=temp_db_path)
    save_message_sqlite("session_B", "user", "Secret for B", db_path=temp_db_path)

    msgs_a = get_recent_messages_sqlite("session_A", db_path=temp_db_path)
    msgs_b = get_recent_messages_sqlite("session_B", db_path=temp_db_path)

    assert len(msgs_a) == 1
    assert msgs_a[0]["content"] == "Secret for A"

    assert len(msgs_b) == 1
    assert msgs_b[0]["content"] == "Secret for B"
