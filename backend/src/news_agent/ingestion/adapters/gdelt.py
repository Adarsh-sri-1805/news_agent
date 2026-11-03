"""Optional GDELT 2.0 Doc API adapter for global and local news discovery."""

from datetime import datetime, timezone
import logging
from typing import Any
import dateutil.parser
import httpx
from pydantic import ValidationError

from news_agent.config import settings
from news_agent.ingestion.adapters.base import BaseNewsSourceAdapter
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now

logger = logging.getLogger(__name__)


class GDELTSourceAdapter(BaseNewsSourceAdapter):
    """Adapter for discovering articles via the GDELT 2.0 Doc API."""

    GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        query: str | None = None,
        max_records: int | None = None,
        timespan: str = "24h",
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        user_agent: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.query = query or settings.gdelt_query
        self.max_records = max_records or settings.gdelt_max_records
        self.timespan = timespan
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
        return f"GDELT:{self.query[:30]}"

    def close(self) -> None:
        """Close HTTP client session if owned."""
        if self._owned_client and not self._client.is_closed:
            self._client.close()

    def fetch_articles(self) -> list[NormalizedArticle]:
        """Query GDELT Doc API and normalize retrieved articles."""
        if not self.query.strip():
            logger.info("GDELT adapter skipped: query is empty.")
            return []

        params = {
            "query": self.query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(self.max_records),
            "timespan": self.timespan,
        }

        data = self._fetch_json(params)
        if not data:
            return []

        articles_data = data.get("articles", [])
        return self._normalize_gdelt_articles(articles_data)

    def _fetch_json(self, params: dict[str, str]) -> dict[str, Any] | None:
        """Fetch GDELT JSON with bounded retries."""
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self._client.get(
                    self.GDELT_DOC_API_URL,
                    params=params,
                    headers={"User-Agent": self.user_agent},
                )
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as err:
                        logger.warning("GDELT returned non-JSON body: %s", err)
                        return None
                logger.warning(
                    "GDELT returned HTTP %s on attempt %d/%d",
                    response.status_code,
                    attempt,
                    self.max_retries + 1,
                )
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as err:
                logger.warning(
                    "GDELT request error (attempt %d/%d): %s",
                    attempt,
                    self.max_retries + 1,
                    err,
                )

        logger.error("Failed to query GDELT API after %d attempts.", self.max_retries + 1)
        return None

    @classmethod
    def _normalize_gdelt_articles(cls, raw_articles: list[dict[str, Any]]) -> list[NormalizedArticle]:
        """Normalize GDELT article dicts to NormalizedArticle models."""
        normalized: list[NormalizedArticle] = []

        for item in raw_articles:
            url = item.get("url")
            title = (item.get("title") or "").strip()
            if not url or not title:
                continue

            publisher = item.get("domain") or item.get("sourcecountry") or "GDELT Source"
            seendate_str = item.get("seendate")
            published_at = cls._parse_gdelt_date(seendate_str)

            # GDELT artlist does not provide full body content; use title / metadata
            content = title
            summary = item.get("socialimage") or None

            try:
                article = NormalizedArticle(
                    title=title,
                    content=content,
                    source_url=url,  # type: ignore[arg-type]
                    publisher=publisher,
                    published_at=published_at,
                    retrieved_at=utc_now(),
                    summary=summary,
                    raw_metadata={
                        "source": "gdelt",
                        "seendate": seendate_str,
                        "sourcecountry": item.get("sourcecountry"),
                        "language": item.get("language"),
                    },
                )
                normalized.append(article)
            except ValidationError as err:
                logger.debug("Validation error normalizing GDELT item '%s': %s", title, err)
                continue

        return normalized

    @staticmethod
    def _parse_gdelt_date(date_str: str | None) -> datetime | None:
        """Parse GDELT format YYYYMMDDTHHMMSSZ or fallback to dateutil."""
        if not date_str:
            return None
        # Standard GDELT format: '20261026T143000Z'
        try:
            dt = datetime.strptime(date_str.strip(), "%Y%m%dT%H%M%SZ")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

        try:
            dt = dateutil.parser.parse(date_str)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None
