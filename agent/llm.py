"""OpenAI LLM client integration and execution.

This module manages secure client initialization, environment configuration,
and error handling for OpenAI Chat Completions API calls.
"""

import os
from typing import Any
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass
import openai
from openai import OpenAI


class LLMError(Exception):
    """Custom application exception for LLM invocation failures."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.original_error = original_error


def _get_client() -> OpenAI:
    """Create and return an OpenAI client instance.

    Raises
    ------
    LLMError
        If OPENAI_API_KEY is not configured in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError(
            "OPENAI_API_KEY is not set. Please set it in your environment or .env file."
        )
    return OpenAI(api_key=api_key)


def call_llm(messages: list[dict[str, str]]) -> str:
    """Invoke the OpenAI Chat Completions API with the supplied messages context.

    Parameters
    ----------
    messages : list[dict[str, str]]
        List of message dicts formatted with 'role' and 'content'.

    Returns
    -------
    str
        The generated assistant response text.

    Raises
    ------
    LLMError
        If API connection fails, rate limits are exceeded, or an API error occurs.
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    # Safeguard against accidentally pasting an API key into the OPENAI_MODEL field
    if not model or model.startswith("sk-"):
        model = "gpt-4o-mini"
    client = _get_client()

    try:
        response: Any = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
        )
        content = response.choices[0].message.content
        return content if content is not None else ""
    except openai.APIConnectionError as err:
        raise LLMError(
            "Could not connect to OpenAI API. Please check your network connection.",
            original_error=err,
        ) from err
    except openai.RateLimitError as err:
        raise LLMError(
            "OpenAI API rate limit or quota exceeded. Please check your plan or retry shortly.",
            original_error=err,
        ) from err
    except openai.APIStatusError as err:
        raise LLMError(
            f"OpenAI API returned an error status (HTTP {err.status_code}): {err.message}",
            original_error=err,
        ) from err
    except openai.OpenAIError as err:
        raise LLMError(
            f"OpenAI client error: {str(err)}",
            original_error=err,
        ) from err
    except LLMError:
        raise
    except Exception as err:
        raise LLMError(
            f"Unexpected error during LLM invocation: {str(err)}",
            original_error=err,
        ) from err
