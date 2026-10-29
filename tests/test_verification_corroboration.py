"""Unit tests for source independence auditing and publisher corroboration analysis."""

from datetime import datetime, timezone
import pytest

from news_agent.models.evidence import EvidencePassage
from news_agent.verification.corroboration import (
    CORROBORATION_LIMITATIONS_NOTE,
    analyze_corroboration,
)


def make_passage(id_val: str, publisher: str, text: str = "Evidence statement.") -> EvidencePassage:
    return EvidencePassage(
        id=id_val,
        passage_text=text,
        publisher=publisher,
        source_url=f"https://{publisher.lower().replace(' ', '')}.com/story",  # type: ignore[arg-type]
        retrieved_at=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
        relevance_score=0.9,
    )


class TestCorroborationAnalysis:
    def test_single_publisher_multiple_passages_not_multi_source(self):
        """CRITICAL: Multiple URLs/passages from the SAME publisher must NOT count as independent corroboration."""
        p1 = make_passage("p1", "Springfield Daily", "Council vote count was 7-2.")
        p2 = make_passage("p2", "Springfield Daily", "Councilors debated the solar issue for two hours.")

        audit = analyze_corroboration([p1, p2])
        assert audit.total_passages == 2
        assert audit.distinct_publisher_count == 1
        assert audit.is_multi_source is False
        assert audit.distinct_publishers == ["Springfield Daily"]

    def test_distinct_publishers_identified_as_multi_source(self):
        p1 = make_passage("p1", "Springfield Daily", "Council vote was 7-2 on Tuesday.")
        p2 = make_passage("p2", "Midwest Chronicle", "City hall approved the solar contract 7-2.")

        audit = analyze_corroboration([p1, p2])
        assert audit.total_passages == 2
        assert audit.distinct_publisher_count == 2
        assert audit.is_multi_source is True
        assert audit.distinct_publishers == ["Midwest Chronicle", "Springfield Daily"]

    def test_syndication_detection_across_publishers(self):
        repeated_text = "Midwest Solar Dynamics won the $2.5 million municipal bidding process."
        p1 = make_passage("p1", "Publisher A", repeated_text)
        p2 = make_passage("p2", "Publisher B", repeated_text)

        audit = analyze_corroboration([p1, p2])
        assert audit.is_multi_source is True
        assert audit.has_potential_syndication is True

    def test_empty_evidence_audit(self):
        audit = analyze_corroboration([])
        assert audit.total_passages == 0
        assert audit.distinct_publisher_count == 0
        assert audit.is_multi_source is False
        assert audit.limitations == CORROBORATION_LIMITATIONS_NOTE
