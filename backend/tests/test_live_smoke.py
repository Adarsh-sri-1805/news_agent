"""Optional live-source smoke test.

This test requires active internet connectivity and is excluded from default unit tests.
To execute: pytest -m live tests/test_live_smoke.py
"""

import pytest
from news_agent.config import FeedSourceConfig
from news_agent.ingestion.adapters.rss import RSSFeedAdapter


@pytest.mark.live
def test_live_rss_feed_smoke():
    """Optional smoke test against live public feed (BBC World News)."""
    config = FeedSourceConfig(
        url="https://feeds.bbci.co.uk/news/world/rss.xml",
        publisher="BBC News",
        name="Live BBC Test",
    )
    adapter = RSSFeedAdapter(feed_config=config, timeout_seconds=10.0, max_retries=1)
    articles = adapter.fetch_articles()

    # If internet is reachable, articles should parse successfully
    assert isinstance(articles, list)
    if articles:
        assert len(articles) > 0
        assert articles[0].publisher == "BBC News"
        assert articles[0].published_at is not None
