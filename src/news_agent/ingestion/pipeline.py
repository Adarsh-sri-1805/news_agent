"""News ingestion pipeline orchestrating adapters, filtering, and deduplication."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any

from news_agent.config import FeedSourceConfig, settings
from news_agent.ingestion.adapters.base import BaseNewsSourceAdapter
from news_agent.ingestion.adapters.gdelt import GDELTSourceAdapter
from news_agent.ingestion.adapters.rss import RSSFeedAdapter
from news_agent.ingestion.dedup import DeduplicationReport, deduplicate_articles
from news_agent.ingestion.filters import filter_by_lookback
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Consolidated result returned by the ingestion pipeline."""

    articles: list[NormalizedArticle]
    raw_collected_count: int
    after_filter_count: int
    deduplication_report: DeduplicationReport
    source_errors: list[dict[str, Any]] = field(default_factory=list)
    ingested_at: datetime = field(default_factory=utc_now)


class IngestionPipeline:
    """Orchestrator for collecting, filtering, and deduplicating news articles."""

    def __init__(
        self,
        adapters: list[BaseNewsSourceAdapter] | None = None,
        lookback_hours: int | None = None,
        title_similarity_threshold: float = 0.90,
    ) -> None:
        self.lookback_hours = (
            lookback_hours if lookback_hours is not None else settings.ingestion_lookback_hours
        )
        self.title_similarity_threshold = title_similarity_threshold
        self.adapters = adapters if adapters is not None else self._build_default_adapters()

    @classmethod
    def _build_default_adapters(cls) -> list[BaseNewsSourceAdapter]:
        """Build adapters according to current application settings."""
        adapters: list[BaseNewsSourceAdapter] = []
        for feed in settings.rss_feeds:
            adapters.append(RSSFeedAdapter(feed_config=feed))

        if settings.gdelt_enabled:
            adapters.append(GDELTSourceAdapter(query=settings.gdelt_query))

        return adapters

    def run(self, reference_time: datetime | None = None) -> IngestionResult:
        """Execute news collection across all configured sources.

        1. Fetches articles from each adapter (safeguarding against adapter crashes).
        2. Applies lookback time-window filtering.
        3. Executes URL canonicalization and conservative title deduplication.
        4. Compiles and returns comprehensive IngestionResult.
        """
        all_raw_articles: list[NormalizedArticle] = []
        source_errors: list[dict[str, Any]] = []

        # 1. Ingest from sources
        for adapter in self.adapters:
            try:
                articles = adapter.fetch_articles()
                logger.info("Adapter '%s' collected %d articles", adapter.source_name, len(articles))
                all_raw_articles.extend(articles)
            except Exception as err:
                logger.exception("Error executing adapter '%s': %s", adapter.source_name, err)
                source_errors.append(
                    {
                        "source": adapter.source_name,
                        "error": str(err),
                        "timestamp": utc_now().isoformat(),
                    }
                )

        raw_count = len(all_raw_articles)

        # 2. Time-window filtering
        filtered_articles = filter_by_lookback(
            all_raw_articles,
            lookback_hours=self.lookback_hours,
            reference_time=reference_time,
        )
        filtered_count = len(filtered_articles)

        # 3. Deduplication
        deduped_articles, dedup_report = deduplicate_articles(
            filtered_articles,
            title_similarity_threshold=self.title_similarity_threshold,
        )

        return IngestionResult(
            articles=deduped_articles,
            raw_collected_count=raw_count,
            after_filter_count=filtered_count,
            deduplication_report=dedup_report,
            source_errors=source_errors,
        )
