"""Unit tests for Pydantic domain models and contracts."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from news_agent.models.article import NormalizedArticle
from news_agent.models.base import ensure_utc
from news_agent.models.brief import NewsBrief
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage
from news_agent.models.social import SocialMediaDraft
from news_agent.models.verification import VerificationResult, VerificationStatus


class TestNormalizedArticle:
    def test_valid_article(self, sample_article: NormalizedArticle):
        assert sample_article.id == "art-12345"
        assert sample_article.publisher == "Springfield Gazette"
        assert str(sample_article.source_url) == "https://springfield-gazette.com/articles/solar-initiative"
        assert sample_article.published_at.tzinfo == timezone.utc
        assert sample_article.retrieved_at.tzinfo == timezone.utc

    def test_naive_datetime_normalized_to_utc(self):
        naive_dt = datetime(2026, 1, 15, 10, 0, 0)
        article = NormalizedArticle(
            title="Headline",
            content="Article text",
            source_url="https://example.com/story",
            publisher="Example News",
            published_at=naive_dt,
        )
        assert article.published_at.tzinfo is not None
        assert article.published_at.tzinfo == timezone.utc

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            NormalizedArticle(
                title="Only Title"
                # missing content, source_url, publisher
            )

    def test_empty_string_validation(self):
        with pytest.raises(ValidationError):
            NormalizedArticle(
                title="",
                content="Valid content",
                source_url="https://example.com",
                publisher="Valid Publisher",
            )

    def test_invalid_url_raises(self):
        with pytest.raises(ValidationError):
            NormalizedArticle(
                title="Valid Title",
                content="Valid Content",
                source_url="not-a-valid-url",
                publisher="Valid Publisher",
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            NormalizedArticle(
                title="Valid Title",
                content="Valid Content",
                source_url="https://example.com",
                publisher="Valid Publisher",
                unrecognized_field="extra_value",
            )


class TestFactualClaim:
    def test_valid_claim(self, sample_claim: FactualClaim):
        assert sample_claim.id == "claim-987"
        assert sample_claim.article_id == "art-12345"
        assert sample_claim.extraction_confidence == 0.95
        assert sample_claim.extracted_at.tzinfo == timezone.utc

    def test_confidence_boundary_validation(self):
        # Confidence out of [0.0, 1.0] must raise
        with pytest.raises(ValidationError):
            FactualClaim(
                article_id="art-1",
                claim_text="Some claim",
                extraction_confidence=1.2,
            )
        with pytest.raises(ValidationError):
            FactualClaim(
                article_id="art-1",
                claim_text="Some claim",
                extraction_confidence=-0.1,
            )

    def test_missing_claim_text_raises(self):
        with pytest.raises(ValidationError):
            FactualClaim(article_id="art-1", claim_text="")


class TestEvidencePassage:
    def test_valid_evidence(self, sample_evidence: EvidencePassage):
        assert sample_evidence.id == "evi-555"
        assert sample_evidence.relevance_score == 0.98
        assert sample_evidence.retrieved_at.tzinfo == timezone.utc

    def test_relevance_boundary_validation(self):
        with pytest.raises(ValidationError):
            EvidencePassage(
                passage_text="Sample text",
                relevance_score=1.5,
            )
        with pytest.raises(ValidationError):
            EvidencePassage(
                passage_text="Sample text",
                relevance_score=-0.5,
            )


class TestVerificationResult:
    def test_explicit_statuses_enum(self):
        assert VerificationStatus.SUPPORTED.value == "supported"
        assert VerificationStatus.CONTRADICTED.value == "contradicted"
        assert VerificationStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"

    def test_valid_verification_result(self, sample_evidence: EvidencePassage):
        result = VerificationResult(
            claim_id="claim-987",
            status=VerificationStatus.SUPPORTED,
            confidence_score=0.92,
            reasoning="Official council record corroborates the 7-2 vote count.",
            evidence_passages=[sample_evidence],
            model_name="llama3.2:3b",
        )
        assert result.status == VerificationStatus.SUPPORTED
        # Verify status is decoupled from confidence score
        assert result.confidence_score == 0.92
        assert len(result.evidence_passages) == 1
        assert result.verified_at.tzinfo == timezone.utc

    def test_invalid_status_value_raises(self):
        with pytest.raises(ValidationError):
            VerificationResult(
                claim_id="claim-987",
                status="partially_true",  # invalid status
                confidence_score=0.8,
                reasoning="Invalid status string",
                model_name="llama3.2:3b",
            )

    def test_confidence_score_boundaries(self):
        with pytest.raises(ValidationError):
            VerificationResult(
                claim_id="claim-987",
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence_score=1.5,
                reasoning="Too confident",
                model_name="llama3.2:3b",
            )


class TestNewsBrief:
    def test_valid_news_brief(self):
        brief = NewsBrief(
            title="Springfield Council Clean Energy Shift",
            headline_takeaways=[
                "City council voted 7-2 for solar arrays on municipal rooftops.",
                "Contractor bidding commences next quarter.",
            ],
            synthesized_body="A comprehensive initiative was passed today by Springfield council...",
            source_article_ids=["art-12345", "art-67890"],
            verification_summary={"supported": 2, "contradicted": 0, "insufficient_evidence": 0},
        )
        assert brief.title == "Springfield Council Clean Energy Shift"
        assert len(brief.headline_takeaways) == 2
        assert brief.verification_summary["supported"] == 2
        assert brief.created_at.tzinfo == timezone.utc

    def test_empty_synthesized_body_raises(self):
        with pytest.raises(ValidationError):
            NewsBrief(
                title="Title",
                synthesized_body="",
            )


class TestSocialMediaDraft:
    def test_character_count_auto_computed(self):
        content = "Breaking: Springfield City Council approves 7-2 solar initiative! #CleanEnergy"
        draft = SocialMediaDraft(
            platform="x_twitter",
            content=content,
            source_article_ids=["art-12345"],
            hashtags=["CleanEnergy", "Springfield"],
        )
        assert draft.character_count == len(content)
        assert draft.platform == "x_twitter"
        assert draft.created_at.tzinfo == timezone.utc

    def test_explicit_character_count_preserved(self):
        draft = SocialMediaDraft(
            platform="linkedin",
            content="Test post",
            character_count=9,
        )
        assert draft.character_count == 9

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            SocialMediaDraft(platform="x_twitter", content="")
