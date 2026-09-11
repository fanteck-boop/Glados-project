# Ollama/Groq-based LLM client with file-context injection.

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

from . import config as cfg

logger = logging.getLogger("voice_bot.llm")


def _system_prompt_with_time(base_prompt: str) -> str:
    """Append the live date/time so the LLM can answer time-based questions
    accurately instead of guessing from training data."""
    now = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    return f"{base_prompt}\n\nCurrent date and time: {now}."


class OllamaClient:
    """Chat client for a local Ollama instance.

    Sends messages to `/api/chat` and returns the assistant's text response.
    Supports a sliding-window conversation history.
    """

    def __init__(self, config: cfg.Config):
        self._config = config
        self._host: str = config.OLLAMA["host"].rstrip("/")
        self._model: str = config.OLLAMA["model"]
        self._max_tokens: int = config.OLLAMA["max_tokens"]
        self._temperature: float = config.OLLAMA["temperature"]
        self._system_prompt: str = config.LLM["system_prompt"]
        self._max_history: int = config.LLM["max_history_turns"]

        # Conversation history as a list of {"role": ..., "content": ...}
        self._history: list[dict[str, str]] = []
        self._lock = __import__("threading").Lock()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self._host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception as e:
            logger.debug("Ollama ping failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, user_text: str) -> str:
        """Send a single user message and return the assistant's response."""
        with self._lock:
            self._history.append({"role": "user", "content": user_text})
            self._trim_history()
            messages = self._build_messages()
        return self._request_chat(messages)

    def chat_with_context(self, user_text: str, file_context: Optional[str]) -> str:
        """Send a user message plus optional file context, return response."""
        with self._lock:
            msg = user_text
            if file_context:
                # Prepend file context as part of the user message so the LLM knows
                msg = f"[CONTEXT FROM FILES]\n{file_context}\n\n[END CONTEXT]\n\n{user_text}"
            self._history.append({"role": "user", "content": msg})
            self._trim_history()
            messages = self._build_messages()
        return self._request_chat(messages)

    def reset(self) -> None:
        with self._lock:
            self._history.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        if len(self._history) > self._max_history:
            # Keep system prompt plus the most recent turns
            keep = self._max_history
            self._history = self._history[-keep:]

    def _build_messages(self) -> list[dict[str, str]]:
        system_content = _system_prompt_with_time(self._system_prompt)
        msgs: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        msgs.extend(self._history)
        return msgs

    def _request_chat(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self._max_tokens,
                "temperature": self._temperature,
            },
        }
        url = f"{self._host}/api/chat"
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            body = r.json()
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "Sorry, the LLM took too long to respond."
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self._host)
            return "Sorry, I can't reach the LLM right now. Is Ollama running?"
        except Exception as e:
            logger.exception("Ollama request error")
            return f"Sorry, something went wrong talking to the LLM: {e}"

        # `body` shape: {"message": {"role":"assistant","content":"..."}, "done":true, ...}
        msg = body.get("message", {})
        content = msg.get("content") or ""
        # Update history with assistant response
        with self._lock:
            self._history.append({"role": "assistant", "content": content})
            self._trim_history()
        return content

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------

    def model_name(self) -> str:
        return self._model


class GroqClient:
    """Chat client for the Groq API (OpenAI-compatible `/chat/completions`).

    Same interface as OllamaClient so it's a drop-in replacement.
    """

    def __init__(self, config: cfg.Config):
        self._config = config
        self._host: str = config.GROQ["host"].rstrip("/")
        self._model: str = config.GROQ["model"]
        self._max_tokens: int = config.GROQ["max_tokens"]
        self._temperature: float = config.GROQ["temperature"]
        self._api_key: str = os.environ.get("GROQ_API_KEY") or config.GROQ.get("api_key", "")
        self._system_prompt: str = config.LLM["system_prompt"]
        self._max_history: int = config.LLM["max_history_turns"]

        self._history: list[dict[str, str]] = []
        self._lock = __import__("threading").Lock()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        if not self._api_key:
            return False
        try:
            r = requests.get(
                f"{self._host}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=5,
            )
            return r.status_code == 200
        except Exception as e:
            logger.debug("Groq ping failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, user_text: str) -> str:
        with self._lock:
            self._history.append({"role": "user", "content": user_text})
            self._trim_history()
            messages = self._build_messages()
        return self._request_chat(messages)

    def chat_with_context(self, user_text: str, file_context: Optional[str]) -> str:
        with self._lock:
            msg = user_text
            if file_context:
                msg = f"[CONTEXT FROM FILES]\n{file_context}\n\n[END CONTEXT]\n\n{user_text}"
            self._history.append({"role": "user", "content": msg})
            self._trim_history()
            messages = self._build_messages()
        return self._request_chat(messages)

    def reset(self) -> None:
        with self._lock:
            self._history.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        if len(self._history) > self._max_history:
            keep = self._max_history
            self._history = self._history[-keep:]

    def _build_messages(self) -> list[dict[str, str]]:
        system_content = _system_prompt_with_time(self._system_prompt)
        msgs: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        msgs.extend(self._history)
        return msgs

    def _request_chat(self, messages: list[dict[str, str]]) -> str:
        if not self._api_key:
            logger.error("No Groq API key configured")
            return "Sorry, no Groq API key is configured."

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        # Only reasoning-capable models accept this param; plain instruct
        # models (e.g. allam-2-7b) reject the request outright if it's set.
        _REASONING_MODELS = ("gpt-oss", "qwen")
        if any(name in self._model for name in _REASONING_MODELS):
            payload["reasoning_effort"] = "none" if "qwen" in self._model else "low"
        url = f"{self._host}/chat/completions"
        try:
            r = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=60,
            )
            r.raise_for_status()
            body = r.json()
        except requests.exceptions.Timeout:
            logger.error("Groq request timed out")
            return "Sorry, the LLM took too long to respond."
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Groq at %s", self._host)
            return "Sorry, I can't reach the LLM right now."
        except requests.exceptions.HTTPError as e:
            logger.error("Groq request failed: %s — %s", e, getattr(e.response, "text", ""))
            return "Sorry, the LLM request failed."
        except Exception as e:
            logger.exception("Groq request error")
            return f"Sorry, something went wrong talking to the LLM: {e}"

        content = ""
        choices = body.get("choices") or []
        if choices:
            content = choices[0].get("message", {}).get("content") or ""

        with self._lock:
            self._history.append({"role": "assistant", "content": content})
            self._trim_history()
        return content

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------

    def model_name(self) -> str:
        return self._model


def make_llm_client(config: cfg.Config):
    """Factory: returns an OllamaClient or GroqClient based on config.LLM_BACKEND."""
    backend = getattr(config, "LLM_BACKEND", "ollama")
    if backend == "groq":
        return GroqClient(config)
    return OllamaClient(config)
