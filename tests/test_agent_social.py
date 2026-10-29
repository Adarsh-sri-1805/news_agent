"""Unit tests for SocialContentAgent: platform character constraints and factual copy integrity."""

from datetime import datetime, timezone
import pytest

from news_agent.agents.social import SocialContentAgent
from news_agent.models.article import NormalizedArticle
from news_agent.models.brief import NewsBrief
from news_agent.models.social import SocialMediaDraft
from news_agent.models.verification import VerificationResult, VerificationStatus


def make_test_article(id_val: str, title: str, publisher: str = "Gazette") -> NormalizedArticle:
    return NormalizedArticle(
        id=id_val,
        title=title,
        content="Article content regarding municipal developments.",
        publisher=publisher,
        source_url=f"https://example.com/{id_val}",  # type: ignore[arg-type]
        published_at=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
    )


class TestSocialContentAgent:
    def test_generate_drafts_enforces_platform_limits_and_formats(self):
        art = make_test_article("art-1", "City Council Approves $2.5M Solar Panel Installation", "Gazette")
        brief = NewsBrief(
            id="brief-1",
            title="Intelligence Brief: Solar Installation Approved",
            headline_takeaways=["Council approves $2.5M installation [Supported]"],
            synthesized_body="Official records confirm approval.",
            source_article_ids=["art-1"],
            verification_summary={"supported": 1, "contradicted": 0, "insufficient_evidence": 0},
        )
        res_supported = VerificationResult(
            claim_id="c1",
            status=VerificationStatus.SUPPORTED,
            confidence_score=0.95,
            reasoning="Confirmed by minutes.",
            model_name="test_model",
        )

        agent = SocialContentAgent()
        drafts = agent.generate_drafts(
            brief=brief,
            articles=[art],
            verification_results=[res_supported],
            platforms=["x_twitter", "linkedin"],
        )

        assert len(drafts) == 2
        draft_x = next(d for d in drafts if d.platform == "x_twitter")
        draft_li = next(d for d in drafts if d.platform == "linkedin")

        # Deterministic character validation
        assert len(draft_x.content) <= 280
        assert draft_x.character_count == len(draft_x.content)
        assert len(draft_li.content) <= 3000
        assert draft_li.character_count == len(draft_li.content)

        # Hashtags must not contain '#' in the hashtags list
        for draft in drafts:
            assert all(not tag.startswith("#") for tag in draft.hashtags)

    def test_contradicted_claims_kept_out_of_factual_copy(self):
        art = make_test_article("art-1", "Council Member Claims Mayor Embezzled Funds", "Tribune")
        brief = NewsBrief(
            id="brief-2",
            title="Intelligence Brief: Embezzlement Allegations",
            headline_takeaways=[
                "Allegations of embezzlement [Contradicted by external evidence]",
            ],
            synthesized_body="City audits prove no funds missing.",
            source_article_ids=["art-1"],
            verification_summary={"supported": 0, "contradicted": 1, "insufficient_evidence": 0},
        )
        res_contradicted = VerificationResult(
            claim_id="c_bad",
            status=VerificationStatus.CONTRADICTED,
            confidence_score=0.99,
            reasoning="Official forensic audit shows zero discrepancies.",
            model_name="test_model",
        )

        agent = SocialContentAgent()
        drafts = agent.generate_drafts(
            brief=brief,
            articles=[art],
            verification_results=[res_contradicted],
            platforms=["x_twitter", "linkedin"],
        )

        for d in drafts:
            # Must NOT claim mayor embezzled funds as a fact
            assert "Mayor Embezzled" not in d.content
            # Must contain advisory or update notice
            assert "Conflicting evidence" in d.content or "Fact-Check Advisory" in d.content

    def test_validate_draft_raises_when_limit_exceeded(self):
        agent = SocialContentAgent()
        excessive_content = "A" * 300
        draft = SocialMediaDraft(
            platform="x_twitter",
            content=excessive_content,
            character_count=300,
        )
        with pytest.raises(ValueError, match="exceeds limit of 280 characters"):
            agent.validate_draft(draft, max_chars=280)
