"""Context assembly for agent long-horizon memory.

This module is responsible for orchestrating the context payload provided to the LLM.
It integrates:
1. System instructions.
2. Retrieved semantic memories (long-term knowledge).
3. Recent short-term conversation messages.
4. The current user message.

Integration note regarding `get_recent_messages()`:
In `agent/loop.py`, `save_message(session_id, 'user', user_message)` is executed prior
to calling `build_context()`. As a result, if `get_recent_messages()` is implemented by
Person 2 to return all records currently stored in the database, it may already include
the latest user message at the end.
`build_context()` handles both scenarios safely:
- If `get_recent_messages()` returns messages up to the current turn (ending with `user_message`),
  the duplicate is stripped from the history block.
- If `get_recent_messages()` only returns messages prior to the current turn, history is preserved as-is.
In both cases, the current user message appears exactly once in the final assembled context as the last item.
"""

from memory.interface import get_recent_messages, retrieve_relevant_memories

SYSTEM_INSTRUCTION: str = (
    "You are a helpful AI agent with long-horizon memory. Use supplied memories "
    "as background information, but do not invent facts. If memories conflict, "
    "prefer the most recent and reliable information. Answer the current user "
    "request directly."
)


def build_context(
    user_message: str,
    session_id: str,
    recent_limit: int = 10,
    memory_limit: int = 5,
) -> list[dict[str, str]]:
    """Assemble the structured context list for the LLM chat completion.

    The context is assembled strictly in this order:
    1. System instructions.
    2. Relevant long-term memories (if any are retrieved).
    3. Recent conversation messages (prior turns).
    4. The current user message.

    Parameters
    ----------
    user_message : str
        The latest prompt/query submitted by the user.
    session_id : str
        Unique identifier for the active conversation session.
    recent_limit : int, default=10
        Maximum number of recent conversation turns to retrieve.
    memory_limit : int, default=5
        Maximum number of semantic memories to retrieve.

    Returns
    -------
    list[dict[str, str]]
        A list of message dictionaries with 'role' and 'content' keys.
    """
    context: list[dict[str, str]] = []

    # 1. System instructions
    context.append({
        "role": "system",
        "content": SYSTEM_INSTRUCTION,
    })

    # 2. Relevant long-term memories
    memories = retrieve_relevant_memories(
        query=user_message,
        session_id=session_id,
        limit=memory_limit,
    )
    if memories:
        formatted_memories = "\n".join(f"- {mem}" for mem in memories if mem and str(mem).strip())
        if formatted_memories:
            context.append({
                "role": "system",
                "content": f"Relevant background memories:\n{formatted_memories}",
            })

    # 3. Recent conversation messages
    recent_messages = get_recent_messages(
        session_id=session_id,
        limit=recent_limit,
    ) or []

    # Deduplication check: if get_recent_messages() already includes the just-saved current user message,
    # strip it from the recent history block so it is not duplicated.
    history_messages = list(recent_messages)
    if (
        history_messages
        and history_messages[-1].get("role") == "user"
        and history_messages[-1].get("content") == user_message
    ):
        history_messages = history_messages[:-1]

    for msg in history_messages:
        context.append({
            "role": str(msg.get("role", "user")),
            "content": str(msg.get("content", "")),
        })

    # 4. The current user message (appears exactly once at the end)
    context.append({
        "role": "user",
        "content": user_message,
    })

    return context
