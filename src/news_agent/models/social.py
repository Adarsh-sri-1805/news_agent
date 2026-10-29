"""Domain contract for generated social media drafts."""

from datetime import datetime
from typing import Any
from uuid import uuid4
from pydantic import Field, field_validator, model_validator

from news_agent.models.base import NewsAgentBaseModel, ensure_utc, utc_now


class SocialMediaDraft(NewsAgentBaseModel):
    """A formatted publication draft tailored for specific social media platforms."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the social draft.",
    )
    platform: str = Field(
        ...,
        min_length=1,
        description="Target social platform (e.g., 'x_twitter', 'linkedin', 'threads').",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Draft copy formatted for the platform.",
    )
    character_count: int = Field(
        default=0,
        ge=0,
        description="Character count of the content (auto-calculated if 0).",
    )
    brief_id: str | None = Field(
        default=None,
        description="Parent NewsBrief ID if derived from a brief.",
    )
    source_article_ids: list[str] = Field(
        default_factory=list,
        description="Referenced source article IDs.",
    )
    hashtags: list[str] = Field(
        default_factory=list,
        description="Curated hashtags without '#'.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="UTC creation timestamp.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Platform parameters such as thread numbering or tone settings.",
    )

    @field_validator("created_at", mode="after")
    @classmethod
    def validate_utc_timestamps(cls, v: datetime | None) -> datetime | None:
        """Enforce timezone awareness and normalize to UTC."""
        return ensure_utc(v)

    @model_validator(mode="after")
    def compute_character_count(self) -> "SocialMediaDraft":
        """Automatically populate or synchronize character count if 0."""
        if self.character_count == 0:
            self.character_count = len(self.content)
        return self
