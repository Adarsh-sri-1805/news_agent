"""FastAPI dependency injection providers."""

from news_agent.agents.orchestrator import NewsIntelligenceOrchestrator
from news_agent.config import Settings, settings
from news_agent.llm import BaseLLMClient, OllamaClient, create_llm_client


def get_settings() -> Settings:
    """Return application settings."""
    return settings


def get_llm_client() -> BaseLLMClient:
    """Return active LLM inference client instance (Gemini or Ollama)."""
    return create_llm_client()


def get_ollama_client() -> OllamaClient:
    """Return local Ollama inference client instance (legacy / backwards-compatible)."""
    client = create_llm_client(provider="ollama")
    if isinstance(client, OllamaClient):
        return client
    return OllamaClient()


def get_orchestrator() -> NewsIntelligenceOrchestrator:
    """Return default multi-agent NewsIntelligenceOrchestrator wired with active LLM client."""
    client = get_llm_client()
    return NewsIntelligenceOrchestrator(llm_client=client)
