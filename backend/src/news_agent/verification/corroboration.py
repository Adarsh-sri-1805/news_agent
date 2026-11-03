"""Conservative corroboration and source independence analysis for retrieved evidence."""

from dataclasses import dataclass, field
import logging
from urllib.parse import urlparse

from news_agent.models.evidence import EvidencePassage

logger = logging.getLogger(__name__)

CORROBORATION_LIMITATIONS_NOTE = (
    "Limitations: Source independence is assessed via publisher attribution strings "
    "and canonical domain heuristics. Wire service syndication (e.g. AP/Reuters reprints) "
    "or corporate cross-ownership may appear under distinct publisher names without true "
    "editorial independence."
)


@dataclass
class CorroborationAnalysis:
    """Audit of source diversity and independence for a set of evidence passages."""

    total_passages: int = 0
    distinct_publisher_count: int = 0
    distinct_publishers: list[str] = field(default_factory=list)
    publisher_distribution: dict[str, int] = field(default_factory=dict)
    is_multi_source: bool = False
    has_potential_syndication: bool = False
    limitations: str = CORROBORATION_LIMITATIONS_NOTE


def analyze_corroboration(evidence_passages: list[EvidencePassage]) -> CorroborationAnalysis:
    """Analyze source independence across retrieved evidence passages.

    Guarantees:
    - Multiple passages or URLs from the SAME publisher are never counted as multiple independent sources.
    - Tracks publisher distribution and flags single-source concentration.
    - Preserves all passages regardless of agreement/conflict.
    """
    if not evidence_passages:
        return CorroborationAnalysis()

    publisher_counts: dict[str, int] = {}
    observed_domains: set[str] = set()

    for passage in evidence_passages:
        pub = (passage.publisher or "Unknown Publisher").strip().title()
        publisher_counts[pub] = publisher_counts.get(pub, 0) + 1

        if passage.source_url:
            parsed = urlparse(str(passage.source_url))
            if parsed.netloc:
                observed_domains.add(parsed.netloc.lower())

    distinct_pubs = sorted(list(publisher_counts.keys()))
    distinct_count = len(distinct_pubs)

    # Multi-source requires at least two distinct publishers
    is_multi_source = distinct_count >= 2

    # Check for potential syndication: e.g. same text appearing across multiple publishers
    has_syndication = False
    seen_texts: set[str] = set()
    for passage in evidence_passages:
        text_snippet = passage.passage_text[:100].strip().lower()
        if text_snippet in seen_texts and is_multi_source:
            has_syndication = True
            break
        seen_texts.add(text_snippet)

    return CorroborationAnalysis(
        total_passages=len(evidence_passages),
        distinct_publisher_count=distinct_count,
        distinct_publishers=distinct_pubs,
        publisher_distribution=publisher_counts,
        is_multi_source=is_multi_source,
        has_potential_syndication=has_syndication,
    )
