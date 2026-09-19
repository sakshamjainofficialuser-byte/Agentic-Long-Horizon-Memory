from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


def save_message(
    session_id: str,
    role: str,
    content: str,
) -> None:
    """Save a single conversation message to persistent storage.

    To be implemented by Person 2 using SQLite.

    Parameters
    ----------
    session_id : str
        Unique identifier for the conversation session.
    role : str
        The role of the sender ('user', 'assistant', 'system').
    content : str
        The message content text.
    """
    ...


def get_recent_messages(
    session_id: str,
    limit: int = 10,
) -> list[Message]:
    """Retrieve recent conversation messages for a session.

    To be implemented by Person 2 using SQLite.

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
    ...


def retrieve_relevant_memories(
    query: str,
    session_id: str,
    limit: int = 5,
) -> list[str]:
    """Retrieve relevant long-term memories using semantic vector search.

    To be implemented by Person 2 using FAISS / Vector store.

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
    ...
