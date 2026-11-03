"""Configuration management using Pydantic Settings."""

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeedSourceConfig(BaseModel):
    """Configuration for an individual RSS/Atom feed source."""

    url: str = Field(..., description="HTTP(S) URL of the RSS or Atom feed.")
    publisher: str = Field(..., description="Publisher name to attribute articles to.")
    name: str | None = Field(default=None, description="Optional friendly name of the feed.")


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    llm_provider: str = Field(
        default="auto",
        description="Active LLM provider: 'auto' (prefers Gemini if key provided, else Ollama), 'gemini', or 'ollama'.",
    )
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("gemini_api_key", "google_api_key"),
        description="Google Gemini API key for fast cloud inference.",
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash",
        description="Gemini model identifier (e.g. gemini-3.5-flash, gemini-3.5-flash-lite).",
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama server instance.",
    )
    ollama_model: str = Field(
        default="llama3.2:3b",
        description="Local Ollama model identifier suitable for 16GB CPU inference.",
    )
    ollama_timeout_seconds: float = Field(
        default=90.0,
        description="Request timeout in seconds for Ollama inference.",
    )
    ollama_keep_alive: str = Field(
        default="5m",
        description="Keep-alive duration in memory for Ollama model (e.g., 5m, 10m).",
    )

    # Ingestion settings
    rss_feeds: list[FeedSourceConfig] = Field(
        default_factory=lambda: [
            FeedSourceConfig(
                url="https://feeds.bbci.co.uk/news/world/rss.xml",
                publisher="BBC News",
                name="BBC World",
            ),
            FeedSourceConfig(
                url="https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
                publisher="The New York Times",
                name="NYT World",
            ),
            FeedSourceConfig(
                url="https://www.theguardian.com/world/rss",
                publisher="The Guardian",
                name="Guardian World",
            ),
            FeedSourceConfig(
                url="https://feeds.npr.org/1001/rss.xml",
                publisher="NPR News",
                name="NPR Top Stories",
            ),
        ],
        description="List of configured RSS/Atom feed sources.",
    )
    ingestion_lookback_hours: int = Field(
        default=24,
        description="Default lookback window in hours for article ingestion.",
    )
    ingestion_timeout_seconds: float = Field(
        default=15.0,
        description="Network request timeout for feed and article fetching.",
    )
    ingestion_max_retries: int = Field(
        default=2,
        description="Maximum retry attempts on transient network failures.",
    )
    ingestion_user_agent: str = Field(
        default="LocalNewsIntelligenceAgent/1.0 (+https://github.com/local-news-intelligence)",
        description="Descriptive User-Agent header for respectful news collection.",
    )
    gdelt_enabled: bool = Field(
        default=False,
        description="Enable optional GDELT 2.0 discovery source.",
    )
    gdelt_query: str = Field(
        default="local government OR municipal council",
        description="Search query string for GDELT discovery.",
    )
    gdelt_max_records: int = Field(
        default=25,
        description="Maximum records to request from GDELT API.",
    )

    app_env: str = Field(
        default="development",
        description="Runtime environment (development, test, production).",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level.",
    )


# Default singleton instance
settings = Settings()
