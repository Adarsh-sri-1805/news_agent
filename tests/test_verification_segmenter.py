"""Unit tests for article passage segmentation and character offset tracking."""

from datetime import datetime, timezone
import pytest

from news_agent.models.article import NormalizedArticle
from news_agent.verification.segmenter import PassageSegment, segment_article


def make_article_with_content(content: str) -> NormalizedArticle:
    return NormalizedArticle(
        id="art-test-100",
        title="Springfield Council Approves Clean Energy Accord",
        content=content,
        source_url="https://springfield.gov/news/solar",  # type: ignore[arg-type]
        publisher="Springfield Municipal",
        published_at=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
    )


class TestPassageSegmentation:
    def test_short_content_yields_single_passage(self):
        short_text = "The city council voted 7-2 on Tuesday morning to approve solar panel installations."
        article = make_article_with_content(short_text)

        passages = segment_article(article, target_chars=400)
        assert len(passages) == 1
        p = passages[0]
        assert p.passage_id == "art-test-100_p0"
        assert p.article_id == "art-test-100"
        assert p.start_char == 0
        assert p.end_char == len(short_text)
        assert p.text == short_text
        assert p.publisher == "Springfield Municipal"

    def test_multi_paragraph_segmentation_offsets_match_verbatim(self):
        para1 = "The Springfield city council voted 7-2 on Tuesday morning to install solar panels on municipal rooftops across the civic center."
        para2 = "The $2.5 million procurement was awarded to Midwest Solar Dynamics following three months of competitive bidding."
        para3 = "Opponents raised concerns regarding short-term debt financing, but proponents cited projected utility savings of $400,000 annually."
        full_content = f"{para1}\n\n{para2}\n\n{para3}"

        article = make_article_with_content(full_content)
        # Set target_chars smaller to force multiple segments
        passages = segment_article(article, target_chars=180, min_chars=50)
        assert len(passages) >= 2

        # Verify exact character slicing matches verbatim text in full_content
        for p in passages:
            sliced = full_content[p.start_char : p.end_char].strip()
            assert p.text == sliced
            assert p.article_id == "art-test-100"

    def test_empty_content_returns_empty_list(self):
        article = NormalizedArticle.model_construct(
            id="art-test-100",
            content="    ",
            source_url="https://example.com",
            publisher="Test Publisher",
        )
        passages = segment_article(article)
        assert passages == []

    def test_to_evidence_passage_conversion(self):
        text = "City records confirm the contract was approved by a 7-2 vote."
        segment = PassageSegment(
            passage_id="art-test-100_p0",
            article_id="art-test-100",
            text=text,
            start_char=0,
            end_char=len(text),
            source_url="https://springfield.gov/records",
            publisher="City Records Office",
            published_at=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
        )

        evidence = segment.to_evidence_passage(relevance_score=0.89)
        assert evidence.id == "art-test-100_p0"
        assert evidence.passage_text == text
        assert evidence.relevance_score == 0.89
        assert evidence.publisher == "City Records Office"
        assert evidence.metadata["start_char"] == 0
        assert evidence.metadata["end_char"] == len(text)
