"""Tests for URL canonicalization, deduplication, and distinction safeguards."""

from datetime import datetime, timezone
import pytest

from news_agent.ingestion.dedup import (
    calculate_title_similarity,
    canonicalize_url,
    deduplicate_articles,
    normalize_title_for_comparison,
)
from news_agent.models.article import NormalizedArticle


def make_test_article(
    title: str,
    url: str,
    content: str = "Standard content body",
    publisher: str = "Test Publisher",
) -> NormalizedArticle:
    return NormalizedArticle(
        title=title,
        content=content,
        source_url=url,  # type: ignore[arg-type]
        publisher=publisher,
        published_at=datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc),
    )


class TestURLCanonicalization:
    def test_strip_tracking_parameters(self):
        url = "https://Example.COM:443/news/article/?utm_source=twitter&utm_medium=social&id=1234&fbclid=XYZ123#comments"
        canonical = canonicalize_url(url)
        assert canonical == "https://example.com/news/article?id=1234"

    def test_port_stripping_and_lowercasing(self):
        http_url = "HTTP://News.Local:80/story/"
        assert canonicalize_url(http_url) == "http://news.local/story"

    def test_query_parameters_sorted(self):
        url = "https://example.com/item?z=9&a=1&m=5"
        assert canonicalize_url(url) == "https://example.com/item?a=1&m=5&z=9"

    def test_empty_url(self):
        assert canonicalize_url("") == ""


class TestTitleNormalizationAndSimilarity:
    def test_normalize_title_for_comparison(self):
        raw = "Breaking: Springfield City Council Approves Solar Plan - Daily Gazette"
        norm = normalize_title_for_comparison(raw)
        assert norm == "springfield city council approves solar plan"

    def test_identical_titles_yield_1_0(self):
        sim = calculate_title_similarity(
            "City Council Approves New Downtown Library",
            "city council approves new downtown library!",
        )
        assert sim == 1.0


class TestDeduplication:
    def test_exact_url_deduplication_preserves_provenance(self):
        art1 = make_test_article(
            "Council Approves Budget",
            "https://springfield.gov/news/budget-2027?utm_source=feed",
        )
        art2 = make_test_article(
            "Council Approves Budget (Duplicate Feed)",
            "https://springfield.gov/news/budget-2027?fbclid=12345",
        )

        retained, report = deduplicate_articles([art1, art2])
        assert len(retained) == 1
        assert report.exact_url_duplicates_removed == 1
        assert report.retained_count == 1

        # Check provenance
        assert str(art2.source_url) in retained[0].raw_metadata.get("merged_source_urls", [])
        assert retained[0].raw_metadata.get("duplicate_url_count") == 1

    def test_near_duplicate_titles_deduplication(self):
        art1 = make_test_article(
            "City Council votes 7-2 to construct new downtown library",
            "https://springfield-gazette.com/articles/library-approved",
        )
        art2 = make_test_article(
            "Breaking: City Council votes 7-2 to construct new downtown library - Gazette",
            "https://wire-service.com/reposts/springfield-library-vote",
        )

        retained, report = deduplicate_articles([art1, art2])
        assert len(retained) == 1
        assert report.near_duplicate_titles_removed == 1

        # Provenance on retained item
        assert art2.title in retained[0].raw_metadata.get("duplicate_titles_seen", [])
        assert retained[0].raw_metadata.get("near_duplicate_count") == 1

    def test_distinct_articles_with_similar_titles_must_not_be_merged(self):
        """CRITICAL REQUIREMENT: Do not merge distinct articles merely because their titles are similar."""
        art_tax = make_test_article(
            "City Council votes on tax hike",
            "https://citynews.com/council-votes-tax-hike",
        )
        art_school = make_test_article(
            "City Council votes on school budget",
            "https://citynews.com/council-votes-school-budget",
        )

        retained, report = deduplicate_articles([art_tax, art_school])
        assert len(retained) == 2
        assert report.exact_url_duplicates_removed == 0
        assert report.near_duplicate_titles_removed == 0
        assert report.retained_count == 2

    def test_another_distinct_articles_pair_not_merged(self):
        art_police = make_test_article(
            "Springfield police department appoints new chief",
            "https://localnews.org/police-appoints-chief",
        )
        art_fire = make_test_article(
            "Springfield fire department appoints new chief",
            "https://localnews.org/fire-appoints-chief",
        )

        retained, report = deduplicate_articles([art_police, art_fire])
        assert len(retained) == 2
        assert report.retained_count == 2
