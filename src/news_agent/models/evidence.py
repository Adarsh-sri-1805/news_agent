"""Domain contract for evidence passages used in fact verification."""

from datetime import datetime
from typing import Any
from uuid import uuid4
from pydantic import Field, HttpUrl, field_validator

from news_agent.models.base import NewsAgentBaseModel, ensure_utc, utc_now


class EvidencePassage(NewsAgentBaseModel):
    """A textual passage retrieved to support, refute, or contextualize a claim."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the evidence passage.",
    )
    passage_text: str = Field(
        ...,
        min_length=1,
        description="The verbatim or extracted text snippet providing evidence.",
    )
    source_url: HttpUrl | None = Field(
        default=None,
        description="Source URL where the evidence passage was retrieved.",
    )
    publisher: str | None = Field(
        default=None,
        description="Publisher or authority source of the evidence passage.",
    )
    retrieved_at: datetime = Field(
        default_factory=utc_now,
        description="UTC timestamp when the passage was retrieved.",
    )
    relevance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Retrieval relevance or similarity score between 0.0 and 1.0.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata preserving chunk offset, query tokens, etc.",
    )

    @field_validator("retrieved_at", mode="after")
    @classmethod
    def validate_utc_timestamps(cls, v: datetime | None) -> datetime | None:
        """Enforce timezone awareness and normalize to UTC."""
        return ensure_utc(v)
