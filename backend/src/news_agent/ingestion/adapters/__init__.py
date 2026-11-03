"""Export ingestion adapters."""

from news_agent.ingestion.adapters.base import BaseNewsSourceAdapter
from news_agent.ingestion.adapters.gdelt import GDELTSourceAdapter
from news_agent.ingestion.adapters.rss import RSSFeedAdapter

__all__ = [
    "BaseNewsSourceAdapter",
    "RSSFeedAdapter",
    "GDELTSourceAdapter",
]
