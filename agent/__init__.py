"""Agent package providing loop execution, context assembly, and LLM communication."""

from typing import Any
from agent.context import SYSTEM_INSTRUCTION, build_context
from agent.llm import LLMError, call_llm

__all__ = [
    "SYSTEM_INSTRUCTION",
    "build_context",
    "call_llm",
    "LLMError",
    "run_agent",
]


def __getattr__(name: str) -> Any:
    """Lazy-load run_agent to avoid circular import warnings when executing python -m agent.loop."""
    if name == "run_agent":
        from agent.loop import run_agent

        return run_agent
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
