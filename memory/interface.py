"""Memory interface implementation connecting SQLite persistence and FAISS vector search."""

from typing import TypedDict
from memory.retrieval import get_memory_store
from memory.storage import get_recent_messages_sqlite, save_message_sqlite


class Message(TypedDict):
    role: str
    content: str


def save_message(
    session_id: str,
    role: str,
    content: str,
) -> None:
    """Save a single message to persistent SQLite storage and index into FAISS.

    Parameters
    ----------
    session_id : str
        Unique identifier for the conversation session.
    role : str
        The role of the sender ('user', 'assistant', 'system').
    content : str
        The message content text.
    """
    # 1. Persist turn to SQLite
    save_message_sqlite(session_id=session_id, role=role, content=content)

    # 2. Index into FAISS for semantic recall
    if role == "user" and content.strip():
        store = get_memory_store()
        store.add_memory(text=content.strip(), session_id=session_id)


def get_recent_messages(
    session_id: str,
    limit: int = 10,
) -> list[Message]:
    """Retrieve the most recent conversation messages from SQLite in chronological order.

    Parameters
    ----------
    session_id : str
        Unique identifier for the conversation session.
    limit : int, default=10
        Maximum number of recent messages to return.

    Returns
    -------
    list[Message]
        A chronological list of recent messages.
    """
    return get_recent_messages_sqlite(session_id=session_id, limit=limit)


def retrieve_relevant_memories(
    query: str,
    session_id: str,
    limit: int = 5,
) -> list[str]:
    """Retrieve relevant long-term memories using FAISS semantic vector search.

    Parameters
    ----------
    query : str
        The search query (typically the latest user message).
    session_id : str
        Unique identifier for the conversation session.
    limit : int, default=5
        Maximum number of relevant memory snippets to return.

    Returns
    -------
    list[str]
        A list of relevant historical memory snippets.
    """
    store = get_memory_store()
    return store.search(query=query, session_id=session_id, limit=limit)
