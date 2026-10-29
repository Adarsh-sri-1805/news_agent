"""Deterministic deduplication and URL canonicalization for news articles."""

from dataclasses import dataclass, field
import difflib
import logging
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from news_agent.models.article import NormalizedArticle

logger = logging.getLogger(__name__)

# Common tracking / click attribution query parameters that do not alter article content
TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "msclkid",
    "mc_eid",
    "mc_cid",
    "_hsenc",
    "_hsmi",
    "ref",
    "ref_src",
    "source",
    "cmpid",
    "mkt_tok",
    "ito",
}


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL safely without altering content identity.

    - Normalizes scheme and host to lowercase.
    - Removes standard default ports (:80, :443).
    - Removes common tracking query parameters.
    - Sorts remaining query parameters.
    - Strips fragment identifier (#...).
    - Removes trailing slash from path (except root /).
    """
    if not url:
        return ""

    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Strip default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # Normalize path
    path = parsed.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # Strip tracking parameters and sort remaining
    filtered_query_params: list[tuple[str, str]] = []
    if parsed.query:
        for k, v in parse_qsl(parsed.query, keep_blank_values=False):
            if k.lower() not in TRACKING_QUERY_PARAMS:
                filtered_query_params.append((k, v))
        filtered_query_params.sort(key=lambda item: item[0])

    new_query = urlencode(filtered_query_params)

    # Reconstruct URL without fragment
    return urlunparse((scheme, netloc, path, parsed.params, new_query, ""))


def normalize_title_for_comparison(title: str) -> str:
    """Normalize article title for conservative similarity checking."""
    if not title:
        return ""
    lowered = title.lower()

    # Strip common news prefixes like 'breaking:', 'update:', 'analysis:', 'opinion:'
    lowered = re.sub(r"^(?:breaking|update|exclusive|analysis|opinion|watch|live)\s*:\s*", "", lowered)

    # Strip trailing publisher tags like ' - the new york times' or ' | bbc news' (requires whitespace around separator)
    lowered = re.sub(r"\s+[-|–—]\s+[a-z0-9\s.]{2,35}$", "", lowered)

    # Remove punctuation
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    # Collapse whitespace
    return re.sub(r"\s+", " ", cleaned).strip()


def calculate_title_similarity(title_a: str, title_b: str) -> float:
    """Calculate SequenceMatcher similarity between normalized titles."""
    norm_a = normalize_title_for_comparison(title_a)
    norm_b = normalize_title_for_comparison(title_b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()


@dataclass
class DeduplicationReport:
    """Report summarizing the outcome and provenance of a deduplication run."""

    input_count: int = 0
    exact_url_duplicates_removed: int = 0
    near_duplicate_titles_removed: int = 0
    retained_count: int = 0
    removed_details: list[dict[str, Any]] = field(default_factory=list)


def deduplicate_articles(
    articles: list[NormalizedArticle],
    title_similarity_threshold: float = 0.90,
) -> tuple[list[NormalizedArticle], DeduplicationReport]:
    """Perform deterministic deduplication across URLs and near-duplicate titles.

    Preserves full provenance on retained articles:
    - Merged alternative URLs.
    - Duplicate titles observed.
    - Duplicate count tracking.

    Args:
        articles: Input list of NormalizedArticles.
        title_similarity_threshold: Threshold for title similarity (default: 0.90).
            Conservative to guarantee distinct stories are never merged.

    Returns:
        Tuple of (retained_articles, deduplication_report).
    """
    report = DeduplicationReport(input_count=len(articles))
    if not articles:
        return [], report

    # 1. Exact URL deduplication
    url_canonical_map: dict[str, NormalizedArticle] = {}
    url_deduped: list[NormalizedArticle] = []

    for article in articles:
        canonical = canonicalize_url(str(article.source_url))
        if canonical in url_canonical_map:
            retained = url_canonical_map[canonical]
            report.exact_url_duplicates_removed += 1

            # Preserve provenance
            merged_urls = retained.raw_metadata.setdefault("merged_source_urls", [])
            merged_urls.append(str(article.source_url))
            retained.raw_metadata["duplicate_url_count"] = (
                retained.raw_metadata.get("duplicate_url_count", 0) + 1
            )

            report.removed_details.append(
                {
                    "reason": "exact_url_duplicate",
                    "removed_id": article.id,
                    "removed_title": article.title,
                    "retained_id": retained.id,
                    "canonical_url": canonical,
                }
            )
            logger.info(
                "Removed exact URL duplicate: '%s' matches retained article '%s'",
                article.title,
                retained.title,
            )
        else:
            article.raw_metadata["canonical_url"] = canonical
            url_canonical_map[canonical] = article
            url_deduped.append(article)

    # 2. Conservative Title near-duplicate deduplication
    retained_articles: list[NormalizedArticle] = []

    for candidate in url_deduped:
        is_duplicate = False

        for existing in retained_articles:
            sim = calculate_title_similarity(candidate.title, existing.title)

            # High similarity check
            if sim >= title_similarity_threshold:
                # To guarantee distinct articles aren't merged, check word token overlap
                tokens_c = set(normalize_title_for_comparison(candidate.title).split())
                tokens_e = set(normalize_title_for_comparison(existing.title).split())
                jaccard = (
                    len(tokens_c & tokens_e) / len(tokens_c | tokens_e)
                    if (tokens_c and tokens_e)
                    else 0.0
                )

                if jaccard >= 0.80:
                    is_duplicate = True
                    report.near_duplicate_titles_removed += 1

                    # Retain the one with more content or preserve provenance
                    dup_titles = existing.raw_metadata.setdefault("duplicate_titles_seen", [])
                    dup_titles.append(candidate.title)
                    existing.raw_metadata["near_duplicate_count"] = (
                        existing.raw_metadata.get("near_duplicate_count", 0) + 1
                    )
                    merged_ids = existing.raw_metadata.setdefault("merged_article_ids", [])
                    merged_ids.append(candidate.id)

                    report.removed_details.append(
                        {
                            "reason": "near_duplicate_title",
                            "removed_id": candidate.id,
                            "removed_title": candidate.title,
                            "retained_id": existing.id,
                            "retained_title": existing.title,
                            "similarity_ratio": round(sim, 3),
                            "jaccard_overlap": round(jaccard, 3),
                        }
                    )
                    logger.info(
                        "Removed near-duplicate title (sim=%.2f): '%s' -> matches '%s'",
                        sim,
                        candidate.title,
                        existing.title,
                    )
                    break

        if not is_duplicate:
            retained_articles.append(candidate)

    report.retained_count = len(retained_articles)
    return retained_articles, report
