"""News Researcher Agent: responsible for ingestion, lookback filtering, deduplication, and source auditing."""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any

from news_agent.ingestion.dedup import DeduplicationReport
from news_agent.ingestion.pipeline import IngestionPipeline, IngestionResult
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now

logger = logging.getLogger(__name__)


@dataclass
class ResearcherResult:
    """Structured output returned by the News Researcher Agent."""

    articles: list[NormalizedArticle]
    raw_collected_count: int
    after_filter_count: int
    retained_count: int
    deduplication_report: DeduplicationReport
    source_errors: list[dict[str, Any]] = field(default_factory=list)
    ingested_at: datetime = field(default_factory=utc_now)

    @property
    def has_errors(self) -> bool:
        """True if any news sources experienced retrieval failures."""
        return len(self.source_errors) > 0


class NewsResearcherAgent:
    """Specialist agent that coordinates news ingestion, deduplication, and source monitoring."""

    def __init__(self, pipeline: IngestionPipeline | None = None) -> None:
        self.pipeline = pipeline or IngestionPipeline()

    def research_news(
        self,
        lookback_hours: int | None = None,
        reference_time: datetime | None = None,
    ) -> ResearcherResult:
        """Execute news research over configured adapters.

        Args:
            lookback_hours: Optional override for time-window filtering.
            reference_time: Optional reference datetime for reproducible tests.

        Returns:
            ResearcherResult with collected articles, deduplication audit, and source error logs.
        """
        # If lookback override provided, temporarily adjust or re-configure pipeline
        original_lookback = self.pipeline.lookback_hours
        if lookback_hours is not None:
            self.pipeline.lookback_hours = lookback_hours

        try:
            raw_result: IngestionResult = self.pipeline.run(reference_time=reference_time)
        except Exception as err:
            logger.exception("Unexpected pipeline failure during news research: %s", err)
            # Never treat fatal pipeline failure as successful zero-article retrieval
            return ResearcherResult(
                articles=[],
                raw_collected_count=0,
                after_filter_count=0,
                retained_count=0,
                deduplication_report=DeduplicationReport(),
                source_errors=[
                    {
                        "source": "pipeline_orchestrator",
                        "error": str(err),
                        "timestamp": utc_now().isoformat(),
                    }
                ],
            )
        finally:
            self.pipeline.lookback_hours = original_lookback

        # Check for source errors
        if raw_result.source_errors:
            logger.warning(
                "News Researcher encountered %d source failure(s) during collection: %s",
                len(raw_result.source_errors),
                [e.get("source") for e in raw_result.source_errors],
            )

        return ResearcherResult(
            articles=raw_result.articles,
            raw_collected_count=raw_result.raw_collected_count,
            after_filter_count=raw_result.after_filter_count,
            retained_count=raw_result.deduplication_report.retained_count,
            deduplication_report=raw_result.deduplication_report,
            source_errors=raw_result.source_errors,
            ingested_at=raw_result.ingested_at,
        )
