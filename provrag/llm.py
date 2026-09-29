"""Optional LLM layer (Together AI). The app works fully without it."""

from __future__ import annotations

import re
from typing import Protocol

from .config import together_api_key, together_model


class ChatLLM(Protocol):
    model: str

    def complete(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 4000) -> str: ...


class TogetherLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from together import Together  # imported lazily: optional dependency at runtime

        self.model = model or together_model()
        self.client = Together(api_key=api_key or together_api_key())

    def complete(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 4000) -> str:
        # Reasoning models spend part of max_tokens on hidden reasoning, so keep it generous.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        text = strip_thinking(choice.message.content or "")
        if choice.finish_reason == "length":
            raise RuntimeError(f"completion truncated at max_tokens={max_tokens}")
        return text


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks some reasoning models emit; we only show the answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def llm_from_env() -> TogetherLLM | None:
    """Return a Together client if TOGETHER_API_KEY is set, otherwise None."""
    if not together_api_key():
        return None
    try:
        return TogetherLLM()
    except Exception:  # missing package, bad config, ...
        return None
