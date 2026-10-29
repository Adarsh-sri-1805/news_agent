"""FastAPI dependency injection providers."""

from news_agent.agents.orchestrator import NewsIntelligenceOrchestrator
from news_agent.config import Settings, settings
from news_agent.llm.client import OllamaClient


def get_settings() -> Settings:
    """Return application settings."""
    return settings


def get_ollama_client() -> OllamaClient:
    """Return local Ollama inference client instance."""
    return OllamaClient()


def get_orchestrator() -> NewsIntelligenceOrchestrator:
    """Return default multi-agent NewsIntelligenceOrchestrator."""
    return NewsIntelligenceOrchestrator()
