"""Thin wrapper around OpenAI and Anthropic chat completion APIs.

Model strings use the format "provider/model_name":
  - "openai/gpt-4.1"         → OpenAI API
  - "anthropic/claude-sonnet-4" → Anthropic API

Also provides MockModel for deterministic testing without API calls.

Design notes (from CLAUDE.md):
  - No hardcoded model names in library code.
  - All functions that call LLM APIs accept a seed parameter.
  - Required env vars checked before first call, not at import time.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class ModelClient(ABC):
    """Abstract base for model clients."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        seed: int | None = None,
        temperature: float = 1.0,
    ) -> str:
        """Generate a completion from a list of messages.

        Args:
            messages: Chat messages as [{"role": ..., "content": ...}].
            system: System prompt (handled differently per provider).
            seed: Passed to API where supported for reproducibility.
            temperature: Sampling temperature.

        Returns:
            The model's response text.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        ...


class OpenAIClient(ModelClient):
    """OpenAI-compatible API client."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required for OpenAI models. "
                "Set it before running Loom."
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI models. "
                "Install with: pip install openai"
            )
        self._client = OpenAI(api_key=api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        seed: int | None = None,
        temperature: float = 1.0,
    ) -> str:
        self._ensure_client()
        api_messages: list[dict[str, str]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "temperature": temperature,
        }
        if seed is not None:
            kwargs["seed"] = seed

        logger.info("OpenAI API call: model=%s, messages=%d", self._model, len(api_messages))
        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        logger.info("OpenAI response: %d chars", len(content))
        return content

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"


class AnthropicClient(ModelClient):
    """Anthropic API client."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic models. "
                "Set it before running Loom."
            )
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic models. "
                "Install with: pip install anthropic"
            )
        self._client = Anthropic(api_key=api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        seed: int | None = None,
        temperature: float = 1.0,
    ) -> str:
        self._ensure_client()
        # Anthropic API uses a separate 'system' parameter.
        # Seed is not natively supported by Anthropic — we log it but
        # pass temperature for partial reproducibility control.
        logger.info("Anthropic API call: model=%s, messages=%d", self._model, len(messages))
        response = self._client.messages.create(
            model=self._model,
            system=system or "You are a helpful assistant.",
            messages=messages,
            max_tokens=1024,
            temperature=temperature,
        )
        content = response.content[0].text if response.content else ""
        logger.info("Anthropic response: %d chars", len(content))
        return content

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "anthropic"


class MockModel(ModelClient):
    """Deterministic model for testing. No API calls.

    Modes:
      - responses: cycle through canned responses.
      - echo: return the last user message.
      - default: return "[Mock response N]".
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        echo: bool = False,
        name: str = "mock/test-model",
    ) -> None:
        self.responses = responses or []
        self.echo = echo
        self.call_count = 0
        self._name = name

    def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        seed: int | None = None,
        temperature: float = 1.0,
    ) -> str:
        self.call_count += 1
        if self.echo:
            return messages[-1]["content"] if messages else ""
        if self.responses:
            idx = (self.call_count - 1) % len(self.responses)
            return self.responses[idx]
        return f"[Mock response {self.call_count}]"

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def provider(self) -> str:
        return "mock"


def create_client(model_string: str) -> ModelClient:
    """Create a ModelClient from a 'provider/model_name' string.

    Examples:
        create_client("openai/gpt-4.1") → OpenAIClient("gpt-4.1")
        create_client("anthropic/claude-sonnet-4") → AnthropicClient("claude-sonnet-4")
        create_client("mock/test") → MockModel()
    """
    if "/" not in model_string:
        raise ValueError(
            f"Model string must be 'provider/model_name', got '{model_string}'. "
            f"Examples: 'openai/gpt-4.1', 'anthropic/claude-sonnet-4'"
        )
    provider, model_name = model_string.split("/", 1)
    provider = provider.lower()

    if provider == "openai":
        return OpenAIClient(model_name)
    elif provider == "anthropic":
        return AnthropicClient(model_name)
    elif provider == "mock":
        return MockModel(name=model_string)
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: openai, anthropic, mock"
        )
