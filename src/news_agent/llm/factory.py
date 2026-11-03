"""Factory for creating LLM clients (Google Gemini or local Ollama)."""

import os
from typing import Any
import httpx

from news_agent.config import settings
from news_agent.llm.base import BaseLLMClient
from news_agent.llm.client import OllamaClient
from news_agent.llm.gemini_client import GeminiClient


def create_llm_client(
    provider: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    timeout_seconds: float | None = None,
    http_client: httpx.Client | None = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """Create and return an LLM client based on configuration.

    Selection rules:
    1. If provider is explicitly 'gemini': return GeminiClient.
    2. If provider is explicitly 'ollama': return OllamaClient.
    3. If provider is 'auto' (or None):
       - If a Gemini API key is configured (GEMINI_API_KEY or GOOGLE_API_KEY),
         prefers GeminiClient for fast, high-quality cloud execution.
       - Otherwise, falls back to OllamaClient for private local execution.
    """
    resolved_provider = (provider or settings.llm_provider or "auto").lower()
    gemini_key = (
        api_key
        or settings.gemini_api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()

    if resolved_provider == "gemini":
        return GeminiClient(
            api_key=gemini_key,
            model_name=model_name or settings.gemini_model,
            timeout_seconds=timeout_seconds,
            http_client=http_client,
        )

    if resolved_provider == "ollama":
        return OllamaClient(
            model_name=model_name or settings.ollama_model,
            timeout_seconds=timeout_seconds,
            http_client=http_client,
            **kwargs,
        )

    # 'auto' mode: prefer Gemini if an API key is present
    if gemini_key:
        return GeminiClient(
            api_key=gemini_key,
            model_name=model_name or settings.gemini_model,
            timeout_seconds=timeout_seconds,
            http_client=http_client,
        )

    return OllamaClient(
        model_name=model_name or settings.ollama_model,
        timeout_seconds=timeout_seconds,
        http_client=http_client,
        **kwargs,
    )

