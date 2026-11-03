"""Tests for RSS and Atom feed adapter parsing and error handling."""

from datetime import datetime, timezone
import httpx
import pytest

from news_agent.config import FeedSourceConfig
from news_agent.ingestion.adapters.rss import (
    RSSFeedAdapter,
    clean_html_text,
    parse_datetime_to_utc,
)

SAMPLE_RSS_2 = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Springfield Daily</title>
    <link>https://springfielddaily.com</link>
    <description>Local community news</description>
    <item>
      <title>City Council Approves New Downtown Library</title>
      <link>https://springfielddaily.com/news/library-approval?utm_source=rss</link>
      <guid>https://springfielddaily.com/news/library-approval</guid>
      <description><![CDATA[The council voted unanimously to construct a new central library.]]></description>
      <content:encoded><![CDATA[<p>Full article body with detailed budget figures and timelines.</p>]]></content:encoded>
      <pubDate>Mon, 27 Oct 2026 10:30:00 GMT</pubDate>
      <dc:creator>Sarah Jenkins</dc:creator>
      <category>Government</category>
      <category>Civic</category>
    </item>
    <item>
      <title>Water Main Repair Closes 4th Street</title>
      <link>https://springfielddaily.com/news/water-main-repair</link>
      <description>Crews are working on 4th Street following a main break.</description>
      <pubDate>Sun, 26 Oct 2026 15:00:00 -0400</pubDate>
    </item>
  </channel>
</rss>
"""

SAMPLE_ATOM_1 = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>County Observer</title>
  <link href="https://countyobserver.org"/>
  <updated>2026-10-27T12:00:00Z</updated>
  <entry>
    <title type="html">&lt;b&gt;County Fair&lt;/b&gt; Dates Announced for 2027</title>
    <link href="https://countyobserver.org/events/fair-2027" rel="alternate"/>
    <id>urn:uuid:60a76c80-d399-11d9-b93C-0003939e0af6</id>
    <published>2026-10-27T08:15:00Z</published>
    <updated>2026-10-27T09:00:00Z</updated>
    <summary>The annual fair returns August 12-18.</summary>
    <content type="html">&lt;p&gt;Preparations are underway for next year's regional fair.&lt;/p&gt;</content>
    <author>
      <name>Marcus Lee</name>
    </author>
    <category term="Events"/>
  </entry>
</feed>
"""

SAMPLE_MALFORMED_XML = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Broken Feed
    <item>
      <title>Unclosed tag
"""


class TestRSSFeedAdapterParsing:
    def test_parse_rss_2_valid(self):
        articles = RSSFeedAdapter.parse_feed_xml(SAMPLE_RSS_2, default_publisher="Springfield Daily")
        assert len(articles) == 2

        first = articles[0]
        assert first.title == "City Council Approves New Downtown Library"
        assert str(first.source_url) == "https://springfielddaily.com/news/library-approval?utm_source=rss"
        assert first.publisher == "Springfield Daily"
        assert first.author == "Sarah Jenkins"
        assert first.published_at == datetime(2026, 10, 27, 10, 30, tzinfo=timezone.utc)
        assert first.tags == ["Government", "Civic"]
        assert "Full article body" in first.content

        second = articles[1]
        assert second.title == "Water Main Repair Closes 4th Street"
        # -0400 offset: 15:00 -04:00 is 19:00 UTC
        assert second.published_at == datetime(2026, 10, 26, 19, 0, tzinfo=timezone.utc)
        assert second.content == "Crews are working on 4th Street following a main break."

    def test_parse_atom_1_valid(self):
        articles = RSSFeedAdapter.parse_feed_xml(SAMPLE_ATOM_1, default_publisher="County Observer")
        assert len(articles) == 1

        entry = articles[0]
        # HTML tag stripped and unescaped
        assert entry.title == "County Fair Dates Announced for 2027"
        assert str(entry.source_url) == "https://countyobserver.org/events/fair-2027"
        assert entry.author == "Marcus Lee"
        assert entry.published_at == datetime(2026, 10, 27, 8, 15, tzinfo=timezone.utc)
        assert entry.tags == ["Events"]
        assert "Preparations are underway" in entry.content

    def test_parse_malformed_xml_returns_empty(self):
        articles = RSSFeedAdapter.parse_feed_xml(SAMPLE_MALFORMED_XML, default_publisher="Broken")
        assert articles == []

    def test_parse_empty_content_returns_empty(self):
        assert RSSFeedAdapter.parse_feed_xml("", default_publisher="None") == []
        assert RSSFeedAdapter.parse_feed_xml("   ", default_publisher="None") == []

    def test_missing_mandatory_fields_skipped(self):
        xml_missing_url = """<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>No Link Article</title>
              <description>Some description</description>
            </item>
            <item>
              <title>Valid Item</title>
              <link>https://example.com/valid</link>
              <description>Valid content</description>
            </item>
          </channel>
        </rss>
        """
        articles = RSSFeedAdapter.parse_feed_xml(xml_missing_url, default_publisher="Example")
        assert len(articles) == 1
        assert articles[0].title == "Valid Item"


class TestRSSFeedAdapterNetworkAndRetries:
    def test_fetch_articles_successful_request(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get("User-Agent") == "TestAgent/1.0"
            return httpx.Response(200, text=SAMPLE_RSS_2)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        config = FeedSourceConfig(url="https://springfielddaily.com/rss", publisher="Springfield Daily")
        adapter = RSSFeedAdapter(
            feed_config=config,
            http_client=client,
            user_agent="TestAgent/1.0",
        )

        articles = adapter.fetch_articles()
        assert len(articles) == 2

    def test_fetch_articles_with_transient_retry_success(self):
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Network blip", request=request)
            return httpx.Response(200, text=SAMPLE_ATOM_1)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        config = FeedSourceConfig(url="https://countyobserver.org/atom", publisher="County Observer")
        adapter = RSSFeedAdapter(feed_config=config, http_client=client, max_retries=2)

        articles = adapter.fetch_articles()
        assert call_count == 2
        assert len(articles) == 1

    def test_fetch_articles_fails_after_retries(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Server Error")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        config = FeedSourceConfig(url="https://bad-feed.com/rss", publisher="Bad Feed")
        adapter = RSSFeedAdapter(feed_config=config, http_client=client, max_retries=1)

        articles = adapter.fetch_articles()
        assert articles == []


class TestDateAndTextHelpers:
    def test_clean_html_text(self):
        raw = "<p>Breaking: &amp; <b>Special</b> report &quot;Local News&quot;   with extra  spaces.</p>"
        cleaned = clean_html_text(raw)
        assert cleaned == 'Breaking: & Special report "Local News" with extra spaces.'

    def test_parse_datetime_to_utc(self):
        # RFC 822 GMT
        dt1 = parse_datetime_to_utc("Mon, 27 Oct 2026 12:00:00 GMT")
        assert dt1 == datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

        # ISO 8601 UTC
        dt2 = parse_datetime_to_utc("2026-10-27T15:45:00Z")
        assert dt2 == datetime(2026, 10, 27, 15, 45, tzinfo=timezone.utc)

        # ISO 8601 Offset (+05:30)
        dt3 = parse_datetime_to_utc("2026-10-27T17:30:00+05:30")
        assert dt3 == datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

        # Invalid returns None
        assert parse_datetime_to_utc("not a valid date") is None
        assert parse_datetime_to_utc("") is None
