"""Pydantic request and response schemas for the REST API."""

from datetime import datetime
from typing import Any
from pydantic import ConfigDict, Field, field_validator

from news_agent.agents.orchestrator import OrchestratorResult
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import NewsAgentBaseModel, utc_now
from news_agent.models.brief import NewsBrief
from news_agent.models.claim import FactualClaim
from news_agent.models.social import SocialMediaDraft
from news_agent.models.verification import VerificationResult

ALLOWED_PLATFORMS = {"x_twitter", "linkedin", "x", "twitter"}


class HealthResponse(NewsAgentBaseModel):
    """Payload for GET /health endpoint."""

    status: str = Field(default="healthy", description="Service health state.")
    app: str = Field(default="local-news-intelligence-agent", description="Application name.")
    version: str = Field(default="0.1.0", description="Application version.")
    timestamp: datetime = Field(default_factory=utc_now, description="Current UTC timestamp.")


class DependencyDetail(NewsAgentBaseModel):
    """Readiness status for an individual external or internal dependency."""

    status: str = Field(..., description="Readiness state ('ready', 'not_ready', 'unavailable', 'warning').")
    details: dict[str, Any] = Field(default_factory=dict, description="Diagnostic dependency metadata.")


class ReadinessResponse(NewsAgentBaseModel):
    """Payload for GET /ready endpoint."""

    status: str = Field(..., description="Overall readiness status ('ready' or 'not_ready').")
    ready: bool = Field(..., description="Boolean flag indicating whether the agent is ready for requests.")
    dependencies: dict[str, DependencyDetail] = Field(
        default_factory=dict,
        description="Detailed readiness checks per dependency component.",
    )
    timestamp: datetime = Field(default_factory=utc_now, description="Current UTC timestamp.")


class NewsRunRequest(NewsAgentBaseModel):
    """Request payload for POST /api/v1/news/run."""

    lookback_hours: int | None = Field(
        default=None,
        ge=1,
        le=168,
        description="Lookback window in hours (1-168, default: 24). Controls news article freshness.",
    )
    max_articles: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Optional limit on number of ingested articles to process through downstream verification (default: all).",
    )
    platforms: list[str] | None = Field(
        default=None,
        description="Target social media platforms (e.g. ['x_twitter', 'linkedin']).",
    )
    mock_mode: bool = Field(
        default=False,
        description="Run in safe mock/demo mode without requiring a running Ollama daemon or live feeds.",
    )
    articles: list[NormalizedArticle] | None = Field(
        default=None,
        description=(
            "Optional pre-supplied articles. Leave omitted or set to [] to run the live "
            "researcher & news ingestion pipeline against configured sources."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lookback_hours": 24,
                "max_articles": 3,
                "platforms": ["x_twitter", "linkedin"],
                "mock_mode": False,
            }
        }
    )

    @field_validator("platforms")
    @classmethod
    def validate_platforms(cls, platforms: list[str] | None) -> list[str] | None:
        if platforms is None:
            return None
        validated: list[str] = []
        for p in platforms:
            cleaned = p.strip().lower()
            if cleaned not in ALLOWED_PLATFORMS:
                raise ValueError(
                    f"Unsupported platform '{p}'. Allowed platforms: {sorted(ALLOWED_PLATFORMS)}"
                )
            validated.append(cleaned)
        return validated


class NewsRunResponse(NewsAgentBaseModel):
    """Structured response for POST /api/v1/news/run reflecting OrchestratorResult."""

    ingestion_metadata: dict[str, Any] = Field(
        description="Telemetry regarding raw articles ingested, filtered, and deduplicated.",
    )
    articles: list[NormalizedArticle] = Field(
        description="Normalized news articles retained for analysis.",
    )
    claims: list[FactualClaim] = Field(
        description="Atomic factual claims extracted from retained articles.",
    )
    verification_results: list[VerificationResult] = Field(
        description="Discrete verification outcomes with confidence scores and evidence citations.",
    )
    brief: NewsBrief | None = Field(
        default=None,
        description="Executive intelligence brief synthesized from articles and verified claims.",
    )
    social_drafts: list[SocialMediaDraft] = Field(
        default_factory=list,
        description="Channel-tailored promotional copy strictly constrained by length and truth.",
    )
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Non-fatal component or source-level error logs recorded during execution.",
    )
    completed_at: datetime = Field(
        default_factory=utc_now,
        description="UTC completion timestamp.",
    )

    @classmethod
    def from_orchestrator_result(cls, res: OrchestratorResult) -> "NewsRunResponse":
        """Instantiate response from internal domain OrchestratorResult without fabricating fields."""
        return cls(
            ingestion_metadata=res.ingestion_metadata,
            articles=res.articles,
            claims=res.claims,
            verification_results=res.verification_results,
            brief=res.brief,
            social_drafts=res.social_drafts,
            errors=res.errors,
            completed_at=res.completed_at,
        )


class ErrorResponse(NewsAgentBaseModel):
    """Sanitized error payload avoiding sensitive exposure."""

    error: str = Field(..., description="High-level error classification.")
    detail: str = Field(..., description="Sanitized description of the error.")
    timestamp: datetime = Field(default_factory=utc_now, description="Current UTC timestamp.")
