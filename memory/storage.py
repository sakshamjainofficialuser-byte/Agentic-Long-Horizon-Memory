"""SQLite persistent storage layer for conversation logs and sessions."""

import os
import sqlite3
from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


DEFAULT_DB_PATH = os.getenv("SQLITE_DB_PATH", "memory.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create and return a configured SQLite connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize the SQLite database schema if not already created."""
    conn = get_db_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id, timestamp)
        """)
        conn.commit()
    finally:
        conn.close()


def save_message_sqlite(
    session_id: str,
    role: str,
    content: str,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Save a single conversation turn to the SQLite messages table."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO messages (session_id, role, content)
            VALUES (?, ?, ?)
            """,
            (session_id, role, content),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()


def get_recent_messages_sqlite(
    session_id: str,
    limit: int = 10,
    db_path: str = DEFAULT_DB_PATH,
) -> list[Message]:
    """Retrieve the most recent conversation messages for a session in chronological order."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (session_id, limit),
        )
        rows = cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]
    finally:
        conn.close()
