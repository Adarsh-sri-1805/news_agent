"""Shared pytest fixtures and test utilities."""

from datetime import datetime, timezone
import pytest
from news_agent.models.article import NormalizedArticle
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage


@pytest.fixture
def sample_utc_datetime() -> datetime:
    return datetime(2026, 10, 25, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_article(sample_utc_datetime: datetime) -> NormalizedArticle:
    return NormalizedArticle(
        id="art-12345",
        title="Local Council Approves New Downtown Solar Initiative",
        content="The Springfield city council voted 7-2 to install solar arrays on municipal buildings.",
        source_url="https://springfield-gazette.com/articles/solar-initiative",
        publisher="Springfield Gazette",
        published_at=sample_utc_datetime,
        author="Jane Reporter",
        summary="Council votes 7-2 for municipal solar arrays.",
        tags=["local-news", "energy", "city-council"],
        raw_metadata={"page": 1, "section": "Local"},
    )


@pytest.fixture
def sample_claim(sample_utc_datetime: datetime) -> FactualClaim:
    return FactualClaim(
        id="claim-987",
        article_id="art-12345",
        claim_text="The city council voted 7-2 in favor of the solar installation.",
        context="Paragraph 1 detailing the voting outcome.",
        extracted_at=sample_utc_datetime,
        extraction_confidence=0.95,
        metadata={"extractor_version": "v1"},
    )


@pytest.fixture
def sample_evidence(sample_utc_datetime: datetime) -> EvidencePassage:
    return EvidencePassage(
        id="evi-555",
        passage_text="Council voting records confirm resolution #44 passed with 7 ayes and 2 nays.",
        source_url="https://springfield.gov/records/2026-resolutions",
        publisher="Springfield Municipal Government",
        retrieved_at=sample_utc_datetime,
        relevance_score=0.98,
        metadata={"record_id": "res-44"},
    )
