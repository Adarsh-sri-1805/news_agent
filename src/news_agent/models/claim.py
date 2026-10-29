"""Domain contract for factual claims extracted from news articles."""

from datetime import datetime
from typing import Any
from uuid import uuid4
from pydantic import Field, field_validator

from news_agent.models.base import NewsAgentBaseModel, ensure_utc, utc_now


class FactualClaim(NewsAgentBaseModel):
    """An atomic factual statement extracted from an article that requires verification."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the extracted claim.",
    )
    article_id: str = Field(
        ...,
        min_length=1,
        description="ID of the parent NormalizedArticle from which this claim was extracted.",
    )
    claim_text: str = Field(
        ...,
        min_length=1,
        description="The atomic verifiable factual statement.",
    )
    context: str | None = Field(
        default=None,
        description="Surrounding context or paragraph to disambiguate the claim.",
    )
    extracted_at: datetime = Field(
        default_factory=utc_now,
        description="UTC timestamp when the claim was extracted.",
    )
    extraction_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model extraction confidence score between 0.0 and 1.0.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata for extraction debugging or provenance.",
    )

    @field_validator("extracted_at", mode="after")
    @classmethod
    def validate_utc_timestamps(cls, v: datetime | None) -> datetime | None:
        """Enforce timezone awareness and normalize to UTC."""
        return ensure_utc(v)
