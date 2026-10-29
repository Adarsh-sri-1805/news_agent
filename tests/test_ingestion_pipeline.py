"""Tests for lookback filtering, GDELT adapter, and the end-to-end IngestionPipeline."""

from datetime import datetime, timedelta, timezone
import httpx
import pytest

from news_agent.ingestion.adapters.base import BaseNewsSourceAdapter
from news_agent.ingestion.adapters.gdelt import GDELTSourceAdapter
from news_agent.ingestion.filters import filter_by_lookback
from news_agent.ingestion.pipeline import IngestionPipeline, IngestionResult
from news_agent.models.article import NormalizedArticle


def create_article_at(title: str, url: str, published_at: datetime | None) -> NormalizedArticle:
    return NormalizedArticle(
        title=title,
        content="Sample content body text",
        source_url=url,  # type: ignore[arg-type]
        publisher="Test Publisher",
        published_at=published_at,
    )


class TestLookbackFiltering:
    def test_filter_by_lookback_within_and_outside_window(self):
        ref_time = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

        # 2 hours old -> keep (2h <= 24h)
        art_recent = create_article_at(
            "Recent News",
            "https://example.com/recent",
            ref_time - timedelta(hours=2),
        )
        # 30 hours old -> drop (30h > 24h)
        art_old = create_article_at(
            "Old News",
            "https://example.com/old",
            ref_time - timedelta(hours=30),
        )
        # Undated -> keep with flag
        art_undated = create_article_at("Undated News", "https://example.com/undated", None)

        retained = filter_by_lookback(
            [art_recent, art_old, art_undated],
            lookback_hours=24,
            reference_time=ref_time,
            keep_undated=True,
        )

        assert len(retained) == 2
        titles = [a.title for a in retained]
        assert "Recent News" in titles
        assert "Undated News" in titles
        assert "Old News" not in titles
        assert art_undated.raw_metadata.get("undated_retained") is True

    def test_filter_by_lookback_drop_undated(self):
        ref_time = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
        art_undated = create_article_at("Undated News", "https://example.com/undated", None)

        retained = filter_by_lookback(
            [art_undated],
            lookback_hours=24,
            reference_time=ref_time,
            keep_undated=False,
        )
        assert len(retained) == 0


class TestGDELTSourceAdapter:
    def test_gdelt_fetch_and_normalize(self):
        sample_gdelt_json = {
            "articles": [
                {
                    "url": "https://gazette.com/metro/zoning-vote",
                    "title": "Zoning Board Approves Mixed-Use District",
                    "seendate": "20261027T101500Z",
                    "domain": "gazette.com",
                    "sourcecountry": "United States",
                    "language": "English",
                }
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            assert "query" in request.url.query.decode("utf-8")
            return httpx.Response(200, json=sample_gdelt_json)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        adapter = GDELTSourceAdapter(
            query="zoning board",
            http_client=client,
        )

        articles = adapter.fetch_articles()
        assert len(articles) == 1
        art = articles[0]
        assert art.title == "Zoning Board Approves Mixed-Use District"
        assert str(art.source_url) == "https://gazette.com/metro/zoning-vote"
        assert art.publisher == "gazette.com"
        assert art.published_at == datetime(2026, 10, 27, 10, 15, tzinfo=timezone.utc)
        assert art.raw_metadata.get("source") == "gdelt"

    def test_gdelt_api_error_returns_empty(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Service Unavailable")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        adapter = GDELTSourceAdapter(query="test", http_client=client, max_retries=1)
        assert adapter.fetch_articles() == []


class DummyAdapter(BaseNewsSourceAdapter):
    def __init__(self, name: str, articles: list[NormalizedArticle], should_fail: bool = False):
        self._name = name
        self._articles = articles
        self._should_fail = should_fail

    @property
    def source_name(self) -> str:
        return self._name

    def fetch_articles(self) -> list[NormalizedArticle]:
        if self._should_fail:
            raise RuntimeError(f"Connection explosion in {self._name}")
        return self._articles


class TestIngestionPipeline:
    def test_pipeline_integration(self):
        ref_time = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

        art1 = create_article_at("City News", "https://news.com/city?utm_source=feed", ref_time - timedelta(hours=1))
        # Exact duplicate of art1 with different tracking param
        art2 = create_article_at("City News Copy", "https://news.com/city?fbclid=abc", ref_time - timedelta(hours=1))
        # Old article > 24 hours
        art3 = create_article_at("Old News", "https://news.com/old", ref_time - timedelta(hours=36))

        adapter_good = DummyAdapter("GoodFeed", [art1, art2, art3])
        adapter_failing = DummyAdapter("FailingFeed", [], should_fail=True)

        pipeline = IngestionPipeline(
            adapters=[adapter_good, adapter_failing],
            lookback_hours=24,
        )

        result = pipeline.run(reference_time=ref_time)
        assert isinstance(result, IngestionResult)
        assert result.raw_collected_count == 3
        # art3 was filtered out by lookback, leaving 2
        assert result.after_filter_count == 2
        # art2 was deduplicated against art1, leaving 1
        assert len(result.articles) == 1
        assert result.articles[0].title == "City News"
        assert result.deduplication_report.exact_url_duplicates_removed == 1

        # Check adapter error capture
        assert len(result.source_errors) == 1
        assert result.source_errors[0]["source"] == "FailingFeed"
        assert "Connection explosion" in result.source_errors[0]["error"]
