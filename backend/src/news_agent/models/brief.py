"""Domain contract for synthesized intelligence news briefs."""

from datetime import datetime
from typing import Any
from uuid import uuid4
from pydantic import Field, field_validator

from news_agent.models.base import NewsAgentBaseModel, ensure_utc, utc_now


class NewsBrief(NewsAgentBaseModel):
    """An executive intelligence brief synthesized from multiple articles and verified claims."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the synthesized news brief.",
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Brief headline or title.",
    )
    headline_takeaways: list[str] = Field(
        default_factory=list,
        description="Bulleted high-level takeaways or key findings.",
    )
    synthesized_body: str = Field(
        ...,
        min_length=1,
        description="Structured analytical summary or deep-dive text.",
    )
    source_article_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source NormalizedArticles referenced in this brief.",
    )
    verification_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Aggregation of verification verdicts (e.g., {'supported': 3, 'contradicted': 1}).",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="UTC creation timestamp.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Synthesis metadata (e.g. topics, sentiment, prompt parameters).",
    )

    @field_validator("created_at", mode="after")
    @classmethod
    def validate_utc_timestamps(cls, v: datetime | None) -> datetime | None:
        """Enforce timezone awareness and normalize to UTC."""
        return ensure_utc(v)
