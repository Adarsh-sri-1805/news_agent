"""News ingestion and deduplication module."""

from news_agent.ingestion.adapters import (
    BaseNewsSourceAdapter,
    GDELTSourceAdapter,
    RSSFeedAdapter,
)
from news_agent.ingestion.dedup import (
    DeduplicationReport,
    calculate_title_similarity,
    canonicalize_url,
    deduplicate_articles,
    normalize_title_for_comparison,
)
from news_agent.ingestion.filters import filter_by_lookback
from news_agent.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "BaseNewsSourceAdapter",
    "RSSFeedAdapter",
    "GDELTSourceAdapter",
    "canonicalize_url",
    "normalize_title_for_comparison",
    "calculate_title_similarity",
    "deduplicate_articles",
    "DeduplicationReport",
    "filter_by_lookback",
    "IngestionPipeline",
    "IngestionResult",
]
