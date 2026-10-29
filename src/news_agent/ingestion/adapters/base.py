"""Base adapter interface for news sources."""

from abc import ABC, abstractmethod
from news_agent.models.article import NormalizedArticle


class BaseNewsSourceAdapter(ABC):
    """Abstract interface for all news collection adapters."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name or identifier of the source adapter."""

    @abstractmethod
    def fetch_articles(self) -> list[NormalizedArticle]:
        """Fetch and return normalized articles from the source.

        Exceptions during fetch should be handled or raised as appropriate,
        while individual malformed entries should be logged and skipped.
        """
