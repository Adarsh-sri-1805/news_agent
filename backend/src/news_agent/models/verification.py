"""Domain contracts for claim verification statuses and results."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4
from pydantic import Field, field_validator

from news_agent.models.base import NewsAgentBaseModel, ensure_utc, utc_now
from news_agent.models.evidence import EvidencePassage


class VerificationStatus(str, Enum):
    """Explicit verification status verdicts.

    Kept strictly decoupled from model confidence scores.
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class VerificationResult(NewsAgentBaseModel):
    """The outcome of verifying a factual claim against retrieved evidence."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the verification result.",
    )
    claim_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the FactualClaim that was evaluated.",
    )
    status: VerificationStatus = Field(
        ...,
        description="Discrete verification verdict (supported, contradicted, insufficient_evidence).",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Statistical model confidence in [0.0, 1.0], maintained separately from the status verdict.",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Chain-of-thought or explanation rationalizing the verification verdict.",
    )
    evidence_passages: list[EvidencePassage] = Field(
        default_factory=list,
        description="Supporting or contradictory evidence passages referenced in reasoning.",
    )
    verified_at: datetime = Field(
        default_factory=utc_now,
        description="UTC timestamp when verification was executed.",
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Identifier of the model/version that produced this verification.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata (tokens, latency, prompt version).",
    )

    @field_validator("verified_at", mode="after")
    @classmethod
    def validate_utc_timestamps(cls, v: datetime | None) -> datetime | None:
        """Enforce timezone awareness and normalize to UTC."""
        return ensure_utc(v)
