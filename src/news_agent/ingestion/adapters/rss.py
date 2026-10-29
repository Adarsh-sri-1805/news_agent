"""RSS and Atom feed adapter for news ingestion."""

from datetime import datetime, timezone
import html
import logging
import re
from typing import Any
import xml.etree.ElementTree as ET
import dateutil.parser
import httpx
from pydantic import ValidationError

from news_agent.config import FeedSourceConfig, settings
from news_agent.ingestion.adapters.base import BaseNewsSourceAdapter
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now

logger = logging.getLogger(__name__)


def clean_html_text(raw_text: str | None) -> str:
    """Unescape HTML entities, strip markup tags, and normalize whitespace."""
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    # Remove HTML tags
    stripped = re.sub(r"<[^>]+>", " ", unescaped)
    # Collapse multiple whitespace characters into single space
    cleaned = re.sub(r"\s+", " ", stripped).strip()
    return cleaned


def parse_datetime_to_utc(date_str: str | None) -> datetime | None:
    """Parse various RFC-822, RFC-2822, and ISO-8601 date strings to UTC-aware datetime."""
    if not date_str or not date_str.strip():
        return None
    try:
        dt = dateutil.parser.parse(date_str)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError) as err:
        logger.debug("Failed to parse date string '%s': %s", date_str, err)
        return None


class RSSFeedAdapter(BaseNewsSourceAdapter):
    """Adapter for ingesting articles from RSS 2.0 and Atom feeds."""

    def __init__(
        self,
        feed_config: FeedSourceConfig,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        user_agent: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.feed_config = feed_config
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.ingestion_timeout_seconds
        )
        self.max_retries = max_retries if max_retries is not None else settings.ingestion_max_retries
        self.user_agent = user_agent or settings.ingestion_user_agent
        self._owned_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )

    @property
    def source_name(self) -> str:
        return self.feed_config.name or f"RSS:{self.feed_config.publisher}"

    def close(self) -> None:
        """Close HTTP client session if owned."""
        if self._owned_client and not self._client.is_closed:
            self._client.close()

    def fetch_articles(self) -> list[NormalizedArticle]:
        """Fetch feed XML from network and parse into NormalizedArticle models."""
        xml_content = self._fetch_feed_content()
        if not xml_content:
            return []
        return self.parse_feed_xml(xml_content, default_publisher=self.feed_config.publisher)

    def _fetch_feed_content(self) -> str | None:
        """Fetch feed XML with bounded retries."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self._client.get(
                    self.feed_config.url,
                    headers={"User-Agent": self.user_agent},
                )
                if response.status_code == 200:
                    return response.text
                logger.warning(
                    "Feed %s returned HTTP %s on attempt %d/%d",
                    self.feed_config.url,
                    response.status_code,
                    attempt,
                    self.max_retries + 1,
                )
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as err:
                last_error = err
                logger.warning(
                    "Request error fetching feed %s (attempt %d/%d): %s",
                    self.feed_config.url,
                    attempt,
                    self.max_retries + 1,
                    err,
                )

        logger.error(
            "Failed to fetch feed %s after %d attempts. Last error: %s",
            self.feed_config.url,
            self.max_retries + 1,
            last_error,
        )
        return None

    @classmethod
    def parse_feed_xml(cls, xml_text: str, default_publisher: str) -> list[NormalizedArticle]:
        """Parse raw XML text (RSS 2.0 or Atom 1.0) into validated NormalizedArticles."""
        if not xml_text or not xml_text.strip():
            logger.warning("Empty feed content received.")
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as err:
            logger.error("XML parse error on feed content: %s", err)
            return []

        root_tag = root.tag.split("}")[-1].lower()
        if root_tag in ("rss", "rdf"):
            return cls._parse_rss_items(root, default_publisher)
        elif root_tag == "feed":
            return cls._parse_atom_entries(root, default_publisher)
        else:
            # Fallback: inspect children for item or entry
            if root.findall(".//item") or root.findall(".//{*}item"):
                return cls._parse_rss_items(root, default_publisher)
            elif root.findall(".//entry") or root.findall(".//{*}entry"):
                return cls._parse_atom_entries(root, default_publisher)

            logger.error("Unrecognized feed root format: <%s>", root.tag)
            return []

    @classmethod
    def _parse_rss_items(cls, root: ET.Element, default_publisher: str) -> list[NormalizedArticle]:
        """Parse RSS 2.0 / RDF items into NormalizedArticle models."""
        articles: list[NormalizedArticle] = []

        # Find publisher from channel title if available
        channel_title = cls._find_child_text(root, "title", recursive=True)
        publisher = default_publisher or channel_title or "Unknown Publisher"

        # Locate all item nodes regardless of XML namespace
        items = [elem for elem in root.iter() if elem.tag.split("}")[-1].lower() == "item"]

        for item in items:
            article = cls._extract_rss_item(item, publisher)
            if article:
                articles.append(article)

        return articles

    @classmethod
    def _extract_rss_item(cls, item: ET.Element, publisher: str) -> NormalizedArticle | None:
        """Extract a single RSS item, validating against NormalizedArticle."""
        title_raw = cls._find_child_text(item, "title")
        title = clean_html_text(title_raw)

        url = cls._find_child_text(item, "link") or cls._find_child_text(item, "guid")
        if url:
            url = url.strip()

        desc_raw = cls._find_child_text(item, "description")
        content_raw = cls._find_child_text(item, "encoded") or desc_raw
        content = clean_html_text(content_raw)
        summary = clean_html_text(desc_raw) if desc_raw else None

        pub_date_raw = cls._find_child_text(item, "pubdate") or cls._find_child_text(item, "date")
        published_at = parse_datetime_to_utc(pub_date_raw)

        author = cls._find_child_text(item, "creator") or cls._find_child_text(item, "author")
        if author:
            author = clean_html_text(author)

        # Tags
        tags = [
            clean_html_text(elem.text)
            for elem in item.iter()
            if elem.tag.split("}")[-1].lower() == "category" and elem.text
        ]

        if not title or not url or not content:
            logger.debug("Skipping RSS item due to missing title, link, or content: '%s'", title)
            return None

        try:
            return NormalizedArticle(
                title=title,
                content=content,
                source_url=url,  # type: ignore[arg-type]
                publisher=publisher,
                published_at=published_at,
                retrieved_at=utc_now(),
                author=author or None,
                summary=summary or None,
                tags=tags,
                raw_metadata={"feed_type": "rss2", "raw_pub_date": pub_date_raw},
            )
        except ValidationError as err:
            logger.warning("Pydantic validation rejected RSS item '%s': %s", title, err)
            return None

    @classmethod
    def _parse_atom_entries(cls, root: ET.Element, default_publisher: str) -> list[NormalizedArticle]:
        """Parse Atom 1.0 entries into NormalizedArticle models."""
        articles: list[NormalizedArticle] = []

        feed_title = cls._find_child_text(root, "title")
        publisher = default_publisher or feed_title or "Unknown Publisher"

        entries = [elem for elem in root.iter() if elem.tag.split("}")[-1].lower() == "entry"]

        for entry in entries:
            article = cls._extract_atom_entry(entry, publisher)
            if article:
                articles.append(article)

        return articles

    @classmethod
    def _extract_atom_entry(cls, entry: ET.Element, publisher: str) -> NormalizedArticle | None:
        """Extract a single Atom entry, validating against NormalizedArticle."""
        title_raw = cls._find_child_text(entry, "title")
        title = clean_html_text(title_raw)

        # Atom links can be <link href="..." rel="alternate"/>
        url: str | None = None
        for elem in entry.iter():
            if elem.tag.split("}")[-1].lower() == "link":
                href = elem.attrib.get("href")
                rel = elem.attrib.get("rel", "alternate")
                if href and rel in ("alternate", "", None):
                    url = href.strip()
                    break
        if not url:
            # fallback to id if URL format
            entry_id = cls._find_child_text(entry, "id")
            if entry_id and entry_id.startswith(("http://", "https://")):
                url = entry_id.strip()

        summary_raw = cls._find_child_text(entry, "summary")
        content_raw = cls._find_child_text(entry, "content") or summary_raw
        content = clean_html_text(content_raw)
        summary = clean_html_text(summary_raw) if summary_raw else None

        date_raw = (
            cls._find_child_text(entry, "published")
            or cls._find_child_text(entry, "updated")
            or cls._find_child_text(entry, "date")
        )
        published_at = parse_datetime_to_utc(date_raw)

        # Author can be <author><name>...</name></author>
        author: str | None = None
        for elem in entry.iter():
            if elem.tag.split("}")[-1].lower() == "author":
                name = cls._find_child_text(elem, "name")
                if name:
                    author = clean_html_text(name)
                    break

        tags: list[str] = []
        for elem in entry.iter():
            if elem.tag.split("}")[-1].lower() == "category":
                term = elem.attrib.get("term") or elem.text
                if term:
                    tags.append(clean_html_text(term))

        if not title or not url or not content:
            logger.debug("Skipping Atom entry due to missing title, link, or content: '%s'", title)
            return None

        try:
            return NormalizedArticle(
                title=title,
                content=content,
                source_url=url,  # type: ignore[arg-type]
                publisher=publisher,
                published_at=published_at,
                retrieved_at=utc_now(),
                author=author or None,
                summary=summary or None,
                tags=tags,
                raw_metadata={"feed_type": "atom", "raw_pub_date": date_raw},
            )
        except ValidationError as err:
            logger.warning("Pydantic validation rejected Atom entry '%s': %s", title, err)
            return None

    @staticmethod
    def _find_child_text(parent: ET.Element, target_tag_suffix: str, recursive: bool = False) -> str | None:
        """Find direct or recursive child whose local tag name matches target_tag_suffix."""
        iterator = parent.iter() if recursive else list(parent)
        target = target_tag_suffix.lower()
        for child in iterator:
            local_name = child.tag.split("}")[-1].lower()
            if local_name == target and child.text and child.text.strip():
                return child.text
        return None
