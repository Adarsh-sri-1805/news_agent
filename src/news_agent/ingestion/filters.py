"""Filtering utilities for ingested news articles."""

from datetime import datetime, timedelta, timezone
import logging
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now

logger = logging.getLogger(__name__)


def filter_by_lookback(
    articles: list[NormalizedArticle],
    lookback_hours: int = 24,
    reference_time: datetime | None = None,
    keep_undated: bool = True,
) -> list[NormalizedArticle]:
    """Filter articles according to publication time within the lookback window.

    Args:
        articles: Candidate list of NormalizedArticles.
        lookback_hours: Maximum age in hours for retained articles.
        reference_time: Optional reference UTC datetime (defaults to utc_now()).
        keep_undated: Whether to retain articles that lack a publication timestamp.

    Returns:
        Filtered list of NormalizedArticles.
    """
    now = reference_time or utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    cutoff = now - timedelta(hours=lookback_hours)
    retained: list[NormalizedArticle] = []

    for article in articles:
        if article.published_at is not None:
            # Ensure published_at is UTC aware
            pub_time = (
                article.published_at.replace(tzinfo=timezone.utc)
                if article.published_at.tzinfo is None
                else article.published_at.astimezone(timezone.utc)
            )
            if pub_time >= cutoff:
                retained.append(article)
            else:
                logger.debug(
                    "Article '%s' excluded by lookback (%s < cutoff %s)",
                    article.title,
                    pub_time.isoformat(),
                    cutoff.isoformat(),
                )
        else:
            if keep_undated:
                article.raw_metadata["undated_retained"] = True
                retained.append(article)
            else:
                logger.debug("Article '%s' excluded: missing publication timestamp", article.title)

    return retained
