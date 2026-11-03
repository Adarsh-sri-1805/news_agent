"""Domain contract for normalized news articles."""

from datetime import datetime
from typing import Any
from uuid import uuid4
from pydantic import Field, HttpUrl, field_validator

from news_agent.models.base import NewsAgentBaseModel, ensure_utc, utc_now


class NormalizedArticle(NewsAgentBaseModel):
    """Represents a canonical normalized news article ingested from any source."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique canonical identifier for the article.",
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Article headline or title.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Full text body or primary content of the article.",
    )
    source_url: HttpUrl = Field(
        ...,
        description="Canonical URL where the article was published.",
    )
    publisher: str = Field(
        ...,
        min_length=1,
        description="Publisher or publication name (e.g., 'Reuters', 'Associated Press').",
    )
    published_at: datetime | None = Field(
        default=None,
        description="UTC publication timestamp if reported by the source.",
    )
    retrieved_at: datetime = Field(
        default_factory=utc_now,
        description="UTC timestamp when the article was fetched/ingested.",
    )
    author: str | None = Field(
        default=None,
        description="Byline author(s) if available.",
    )
    summary: str | None = Field(
        default=None,
        description="Optional initial summary or deck text.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Topic categories or keywords associated with the article.",
    )
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible source-specific metadata preserved without data loss.",
    )

    @field_validator("published_at", "retrieved_at", mode="after")
    @classmethod
    def validate_utc_timestamps(cls, v: datetime | None) -> datetime | None:
        """Enforce timezone awareness and normalize to UTC."""
        return ensure_utc(v)
